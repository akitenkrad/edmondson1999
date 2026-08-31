//! runvault への記録の共通部分．
//!
//! 論文メタデータ (research) は `run` / `sweep` / `reproduce` のどれでも同一なので，
//! ここ 1 箇所で組み立てる．実験条件・ステップごとの指標・チームの観測行・試行の
//! 終端行もここに集める．
//!
//! # 4 つの旧 CSV をどこへ置いたか
//!
//! 旧 `run` は 1 つのディレクトリに形の違う 4 枚の CSV を並べていた．runvault の
//! `metrics.csv` は (`run_uid`, `step`, `step_unit`, `scope`, `name`) が主キーで
//! 系列 ID の列を持たないので，時刻と系列の 2 つのキーで決まる行はここに載らない．
//! 粒度で分けた:
//!
//! - `metrics.csv` (`t,icc_psi,icc_learning,mediation_ratio,beta_psi_l,beta_l_pi`)
//!   → **`metrics.csv` の `scope=run` ステップ指標**．どれもチーム集合全体を 1 つの
//!   数で表す量で，系列を持たない．列名は旧のまま．
//! - `teams.csv` (`t,team_id,…`)
//!   → **`observation` イベント** (1 ステップ 1 チーム 1 行)．`unit_id` がチーム，
//!   `t` が時刻，残りが欄．チームは «毎ステップ観測される主体» で，zhao2024 の
//!   店舗・han2023 の企業と同じ形である．
//! - `team_cross_section.csv` (`run,team_id,…`)
//!   → **`terminal` イベント** (1 チーム 1 行)．`run` 列は «同一条件の反復» なので
//!   子 run の `replicate_index` になり (下記)，残った `team_id` × 1 行が
//!   チームの終端行になる．値は run 後半の時間平均 (原著の «調査票による安定した
//!   チーム構成概念» の対応物) で，観測行のどれとも別の数である．run 内で一定の
//!   `icc_psi` / `icc_learning` の 2 列は全チーム行に重複していたので，run スコープ
//!   の指標 1 本ずつに移した (`icc_psi` は最終ステップのステップ指標そのものなので
//!   書かない — 同じ数を 2 箇所に置かない)．
//! - `individuals.csv` (`t,team_id,agent_id,…`)
//!   → **`artifacts/individuals.csv` に表のまま**．17,280 行 (720 人 × 24 ステップ)
//!   と重く，1 行が潜在状態 (`psi_i` / `fear`) とその場の行動 (`voice` /
//!   `retaliated`) を組にして持つ．下流の解析はこの表を読まない (Python 側は
//!   teams / metrics / cross-section しか使わない) ので，割ってイベントにしても
//!   読み手が増えない．fujimura2019 の `agent_panel.csv` と同じ扱いである．
//!
//! # 反復 (replicate) の形
//!
//! `run --runs N` は独立な N 本のシミュレーションを回し，旧実装は **最後の 1 本の
//! 時系列だけ** を `teams.csv` / `individuals.csv` / `metrics.csv` に書き，
//! `team_cross_section.csv` にだけ N 本ぶんを積んでいた．反復ごとの時系列は捨てられて
//! いたが，それは 1 ディレクトリ 1 ファイルという置き場の制約から来たものである．
//!
//! **反復 1 本を子 run 1 本にした．** 親が [`RunOptions::sweep_parent`] を名乗って
//! 条件と反復リストを宣言し (反復ごとの派生シードで駆動されるので単一の
//! `master_seed` は名乗らない)，子がそれぞれ自分の `master_seed` と
//! `replicate_index` を持つ．同一条件の反復は `config_hash` が一致するので，
//! «どれとどれが同じ条件の繰り返しか» は機械が答える．
//!
//! 決め手は «反復が時系列を持つか» である．`run` の反復は全ステップを観測しており
//! (旧実装が捨てていただけで)，ステップ系列の置き場は `metrics.csv` しかないので
//! 反復ごとに 1 本要る．一方 `sweep` と `reproduce` が見るのは各試行の最終値だけ
//! なので，試行は自分の時系列を持たず `terminal` 行 1 本で言い尽くせる —
//! こちらは hegselmann2002 / fujimura2019 と同じく «条件 1 点 = 子 run 1 本，
//! 試行 = terminal 行» にした．形は «何を観測したか» に従わせてある．

use runvault::{Llm, Replication, Run, Target, Work};
use serde::Serialize;

use crate::config::{Config, LearningWeights, NetworkKind, PsiParams, VoiceBeta};
use crate::simulation::{AnchorReport, CrossSectionRow, MetricsRow, SimulationResult, TeamRow};

/// runvault 上の実験名．`runvault path --experiment` に渡す値でもある．
pub const EXPERIMENT: &str = "edmondson-psafety";
/// リポジトリの安定 id．git remote の名前とは独立に固定する．
pub const REPO_ID: &str = "edmondson1999";
/// 分野．個人属性・チーム網・スケジューラ・観察者ノイズを乱数で引くので
/// `simulation` (= `master_seed` が必須)．
pub const DOMAIN: &str = "simulation";

/// 時間軸の単位．モデルの刻みは離散 tick なので語彙の `step`．
const T_UNIT: &str = "step";

/// 指標の粒度．ICC も回帰係数もチーム集合全体の集約なので `run`．
const SCOPE: &str = "run";

// ---------------------------------------------------------------------------
// 論文メタデータ
// ---------------------------------------------------------------------------

/// この再現実験が対象としている論文．
///
/// target は 2 つ持つ．原著の主張は «コンテクスト支援・リーダーコーチング →
/// 心理的安全性 → 学習行動 → チームパフォーマンス» という媒介連鎖 (H1–H4, H6, H7)
/// と，«チーム効力感は $\psi$ を統制すると学習行動を説明しない» (H5, H8 不支持) の
/// 2 本で，どちらもこの実装が run 1 本の中で測る量に対応する．
pub fn replication() -> Replication {
    Work::paper_id("P00001816")
        .title("Psychological Safety and Learning Behavior in Work Teams")
        .year(1999)
        .source_version("published")
        .target(Target::claim(
            "psafety-mediates-support-to-performance",
            "Context support and leader coaching raise team psychological safety, which drives learning behavior and thereby team performance",
        ))
        .target(Target::claim(
            "efficacy-is-discriminant",
            "Team efficacy does not explain learning behavior once psychological safety is controlled",
        ))
        .obsidian_note("研究/98_論文レポート/80-再現実験/実装完了/edmondson1999/設計書.md")
}

/// 原著が報告した値のうち，この実装が run 1 本の中で直接測るもの．
///
/// 書けるのは Edmondson (1999) の本文・Table が印字した 8 つの数だけである．
/// 設計書 §5 が «補正アンカー» として並べている帯 ($[.25,.55]$ 等) はこちらが
/// 決めた許容幅なので入れない — 原著が印字した数と自前の帯が後から見分けられなく
/// なる．`mediation_ratio` と `efficacy_partial_t` も書かない (原著は媒介比率も
/// 効力感の偏 $t$ 値も数として報告していない．報告しているのは «残差が有意でない»
/// という判定と，その残差係数 $B=.25\ p=.42$ である)．
///
/// `reproduce` でだけ呼ぶ．run スコープに対応する観測値が無い run に報告値だけを
/// 置くと，比較できない数が記録に残る．
pub fn log_paper_reference(run: &mut Run) {
    const MEDIATION: &str = "psafety-mediates-support-to-performance";
    const DISCRIMINANT: &str = "efficacy-is-discriminant";
    let rows: [(&str, f64, &str, &str); 8] = [
        (
            "icc_psi",
            0.39,
            MEDIATION,
            "Edmondson (1999) Table 3: ICC(team psychological safety) = .39",
        ),
        (
            "icc_learning_stable",
            0.27,
            MEDIATION,
            "Edmondson (1999) Table 3: ICC(team learning behavior) = .27",
        ),
        (
            "beta_psi_l",
            0.76,
            MEDIATION,
            "Edmondson (1999) Table 5A model 1: psychological safety -> learning behavior B = .76 (p < .01)",
        ),
        (
            "r2_psi_l",
            0.63,
            MEDIATION,
            "Edmondson (1999) Table 5A model 1: adjusted R^2 = .63",
        ),
        (
            "r2_l_pi",
            0.26,
            MEDIATION,
            "Edmondson (1999) Table 4: learning behavior -> team performance, adjusted R^2 = .26",
        ),
        (
            "beta_support_psi",
            0.56,
            MEDIATION,
            "Edmondson (1999) Table 6: context support -> psychological safety B = .56 (p < .001)",
        ),
        (
            "beta_psi_residual",
            0.25,
            MEDIATION,
            "Edmondson (1999) Table 4: psychological safety residual after learning behavior, B = .25",
        ),
        (
            "beta_psi_residual_p",
            0.42,
            DISCRIMINANT,
            "Edmondson (1999) Table 4: the same residual is non-significant, p = .42",
        ),
    ];
    for (name, value, target, source) in rows {
        run.log_reference(name, value)
            .target(target)
            .source(source)
            .send()
            .unwrap_or_else(|e| panic!("原著の報告値 `{name}` の記録に失敗: {e}"));
    }
}

// ---------------------------------------------------------------------------
// LLM ブロック
// ---------------------------------------------------------------------------

/// 実際に応答したバックエンドを `llm` ブロックに落とす．
///
/// `model` / `endpoint` はクライアントが名乗った値をそのまま使う．`provider` は
/// runvault の語彙ではなく自由記述なので，endpoint から «どのゲートウェイが
/// 答えたか» を決める．
///
/// rule モードではこれを呼ばない．LLM を 1 回も叩かない run に `llm` ブロックを
/// 付けると，存在しないモデル (旧 `llm_meta.json` の `"none"`) を名乗ることになる．
pub fn llm_block(model: &str, endpoint: &str, temperature: f32) -> Llm {
    let provider = if endpoint.starts_with("mock://") {
        "mock"
    } else if endpoint.contains("openai") {
        "openai"
    } else {
        "ollama"
    };
    Llm {
        provider: provider.to_string(),
        model_snapshot: model.to_string(),
        temperature: Some(temperature as f64),
        // プロンプトは個人ごとに組み立てられ，固定の system prompt を持たない．
        // 無いものを hash しない．
        system_prompt_hash: None,
    }
}

// ---------------------------------------------------------------------------
// 実験条件 (parameters)
// ---------------------------------------------------------------------------

/// 条件そのもの．シードも反復数も出力先も含まない．
///
/// 旧 `config.json` (=`run` のみ) が持っていた項目に加え，`sweep` と `reproduce`
/// で **結果を決めていたのに記録されていなかった** 値をすべて入れてある．旧
/// `sweep_config.json` は $\alpha$ / $\delta$ / $\lambda$ しか書かず，$\beta$ /
/// $\gamma$・網の形・$\sigma_{obs}$・$\gamma_L$ / $\gamma_K$・voice ロジットの係数群
/// などは `Config::default()` から来ていた．`reproduce` に至っては設定ファイルを
/// 1 つも書いていない．どれも結果を決める量なので，落とすと «同じ条件» を名乗る
/// 2 本の run が違う係数で回っていた，ということが起こりうる．
///
/// $\psi$ 更新式の重みだけ平坦に置いてある ($\lambda$ / $\alpha$ / $\beta$ /
/// $\gamma$ / $\delta$)．`sweep` が掃くのがこの群で，`runvault.read.
/// sweep_events_table` が条件列にできるのは `/parameters` の最上位キーだけだから
/// である．掃かない `voice_beta` / `learning_weights` は旧 `config.json` と同じく
/// 入れ子のままにした．
#[derive(Serialize)]
pub struct ConditionParameters {
    pub decision_mode: &'static str,
    pub n_teams: usize,
    pub team_size: usize,
    pub n_individuals: usize,
    pub network_kind: NetworkKind,
    pub network_k: usize,
    pub network_beta: f64,
    /// $\psi$ 更新式の重み．最上位に平坦化する (上記)．
    #[serde(flatten)]
    pub psi: PsiParams,
    pub voice_beta: VoiceBeta,
    pub learning_weights: LearningWeights,
    pub gamma_l: f64,
    pub gamma_k: f64,
    pub sigma_obs: f64,
    pub knowledge_decay: f64,
    pub p_retaliate: f64,
    pub t_max: u64,
    pub llm_temperature: f32,
    pub llm_seed: u64,
    /// プロンプト → 応答キャッシュの置き場．条件ではなく置き場なので
    /// `hash_exclude` で `config_hash` から外す ([`HASH_EXCLUDE`])．
    pub llm_cache_path: Option<String>,
}

/// `config_hash` から外すポインタ．
///
/// キャッシュのパスは «どこに置いたか» であって条件ではない．同じ条件の run を
/// 別のキャッシュファイルで回しても同じ条件である．
pub const HASH_EXCLUDE: [&str; 1] = ["/llm_cache_path"];

impl ConditionParameters {
    /// [`Config`] から条件だけを取り出す．
    pub fn from_config(cfg: &Config) -> Self {
        ConditionParameters {
            decision_mode: cfg.decision_mode.label(),
            n_teams: cfg.n_teams,
            team_size: cfg.team_size,
            n_individuals: cfg.n_individuals(),
            network_kind: cfg.network_kind,
            network_k: cfg.network_k,
            network_beta: cfg.network_beta,
            psi: cfg.psi,
            voice_beta: cfg.voice_beta,
            learning_weights: cfg.learning_weights,
            gamma_l: cfg.gamma_l,
            gamma_k: cfg.gamma_k,
            sigma_obs: cfg.sigma_obs,
            knowledge_decay: cfg.knowledge_decay,
            p_retaliate: cfg.p_retaliate,
            t_max: cfg.t_max,
            llm_temperature: cfg.llm.temperature,
            llm_seed: cfg.llm.seed,
            llm_cache_path: cfg.llm.cache_path.clone(),
        }
    }
}

/// 反復 1 本 (子 run) の実験条件．
///
/// `seed` はこの反復が実際に使ったシードで，`master_seed` と同じ値である．
/// `seed_pointers` で seed として宣言するので `config_hash` からは外れ，
/// 同一条件の反復どうしは `config_hash` が一致する．
#[derive(Serialize)]
pub struct ReplicateParameters {
    #[serde(flatten)]
    pub condition: ConditionParameters,
    pub seed: u64,
}

/// 「1 条件 + その反復群」の実験条件．
///
/// `run` の親 run，`sweep` のセル子 run，`reproduce` の run がこの形をしている．
/// どれも «この条件を `runs` 本回す» という宣言である．`base_seed` は反復ごとの
/// シードを派生させる元で，それ自体でシミュレーションを回す値ではない．
#[derive(Serialize)]
pub struct ReplicateGroupParameters {
    #[serde(flatten)]
    pub condition: ConditionParameters,
    pub runs: usize,
    pub base_seed: u64,
}

/// `run` 親 / `sweep` セル子 / `reproduce` の seed ポインタ．
pub const GROUP_SEED_POINTERS: [&str; 1] = ["/base_seed"];
/// 反復 (子 run) の seed ポインタ．
pub const REPLICATE_SEED_POINTERS: [&str; 1] = ["/seed"];

// ---------------------------------------------------------------------------
// 反復 1 本 (run の子) の記録
// ---------------------------------------------------------------------------

/// `events.jsonl` に書くチームの観測行 (旧 `teams.csv` の 1 行)．
///
/// 予約キー 3 つのあとが自由欄．チーム固有の 6 列はここにしか置けない — 時刻と
/// チームの 2 つのキーで決まる行は `metrics.csv` の主キーを名乗れない．
#[derive(Serialize)]
struct TeamObservation<'a> {
    unit_id: &'a str,
    t: u64,
    t_unit: &'static str,
    psi: f64,
    learning: f64,
    performance: f64,
    efficacy: f64,
    support: f64,
    coaching: f64,
}

/// `events.jsonl` に書くチームの終端行 (旧 `team_cross_section.csv` の 1 行)．
///
/// 先頭 6 フィールドは runvault の予約語 (`terminal` はこれを全部要求する)．
/// 残りは run 後半の時間平均で，観測行のどれとも別の数である (原著のチーム構成概念
/// は単一 tick の値ではなく調査票の安定集約なので，後半平均がその対応物になる)．
///
/// チームは途中で退出しないのでモデルは必ず `t_max` まで回る．`outcome` は常に
/// `horizon`，`censored` は真である (打ち切りの行は `t == budget` でなければ
/// ならず，`final_round == t_max` なのでこれを満たす)．
#[derive(Serialize)]
struct TeamTerminal<'a> {
    unit_id: &'a str,
    t: u64,
    t_unit: &'static str,
    outcome: &'static str,
    censored: bool,
    budget: u64,
    psi: f64,
    learning: f64,
    performance: f64,
    support: f64,
    efficacy: f64,
}

/// 反復 1 本ぶんの記録．
///
/// ステップごとの 5 指標 (`t` は時間軸なので値としては書かない)，チームの観測行と
/// 終端行，run 全体を 1 つの値で表す指標を書く．旧 `metrics.csv` の列名は
/// そのまま使う．実行時間は `status.json` の `duration_sec` が正本なので指標に
/// しない．
pub fn log_replicate(run: &mut Run, cfg: &Config, result: &SimulationResult) {
    for m in &result.metrics_rows {
        log_step(run, m);
    }
    for row in &result.team_rows {
        log_team_observation(run, row);
    }
    for row in &result.cross_section_rows() {
        log_team_terminal(run, row, result.final_round, cfg.t_max);
    }

    let mut scoped: Vec<(&str, f64)> = vec![
        // 観測主体はチーム (個人は `artifacts/individuals.csv` にあるが，
        // 終端を持つ観測単位ではない)．
        ("n_units", cfg.n_teams as f64),
        ("final_round", result.final_round as f64),
        // 旧 `team_cross_section.csv` の `icc_learning` 列．全チーム行に同じ値が
        // 重複していた run スコープの量なので，ここに 1 本だけ書く．ステップ指標の
        // `icc_learning` (その時刻の voiced_last の ICC) とは別の量 — こちらは
        // 個人ごとの時間平均発言率の ICC で，原著 Table 3 の .27 に対応する安定推定
        // なので別名にした．
        ("icc_learning_stable", result.icc_learning),
    ];
    // 収束しなかった run に «収束ステップ 0» を名乗らせない (欠測は行を書かない)．
    if let Some(step) = result.convergence_step {
        scoped.push(("convergence_step", step as f64));
    }
    run.log_metrics(SCOPE, &scoped)
        .expect("run スコープの指標の記録に失敗");
}

/// [`MetricsRow`] の数値フィールドを 1 ステップぶんまとめて書く．
///
/// どれもチーム集合全体を 1 つの数で表す量 (ICC・チーム間 OLS の係数・媒介比率)
/// なので系列を持たず，`scope=run` のステップ指標に載る．
fn log_step(run: &mut Run, m: &MetricsRow) {
    run.log_metrics_at(
        m.t,
        T_UNIT,
        SCOPE,
        &[
            ("icc_psi", m.icc_psi),
            ("icc_learning", m.icc_learning),
            ("mediation_ratio", m.mediation_ratio),
            ("beta_psi_l", m.beta_psi_l),
            ("beta_l_pi", m.beta_l_pi),
        ],
    )
    .unwrap_or_else(|e| panic!("step {} の指標の記録に失敗: {e}", m.t));
}

/// チーム 1 つの 1 ステップぶんの観測行．
fn log_team_observation(run: &mut Run, row: &TeamRow) {
    let unit_id = team_unit_id(row.team_id);
    run.log_event(
        "observation",
        &TeamObservation {
            unit_id: &unit_id,
            t: row.t,
            t_unit: T_UNIT,
            psi: row.psi,
            learning: row.learning,
            performance: row.performance,
            efficacy: row.efficacy,
            support: row.support,
            coaching: row.coaching,
        },
    )
    .unwrap_or_else(|e| panic!("{unit_id} の t={} の observation の記録に失敗: {e}", row.t));
}

/// チーム 1 つの終端行．
fn log_team_terminal(run: &mut Run, row: &CrossSectionRow, final_round: u64, t_max: u64) {
    let unit_id = team_unit_id(row.team_id);
    let censored = final_round == t_max;
    run.log_event(
        "terminal",
        &TeamTerminal {
            unit_id: &unit_id,
            t: final_round,
            t_unit: T_UNIT,
            outcome: if censored { "horizon" } else { "stopped" },
            censored,
            budget: t_max,
            psi: row.psi,
            learning: row.learning,
            performance: row.performance,
            support: row.support,
            efficacy: row.efficacy,
        },
    )
    .unwrap_or_else(|e| panic!("{unit_id} の terminal イベントの記録に失敗: {e}"));
}

/// チームの `unit_id`．
fn team_unit_id(team_id: u32) -> String {
    format!("team-{team_id}")
}

/// LLM 呼び出しの内訳を run スコープの指標として書く．
///
/// rule モードでは呼ばない．0 回という数を書くこと自体は嘘ではないが，LLM を
/// 配線していない run に LLM の指標が並ぶと，`llm` ブロックの有無と食い違って
/// 見える．率は呼び出しが 1 本も無いときに «0» ではなく «定義できない» ので，
/// そのときは行そのものを書かない．
///
/// `tokens_in` / `tokens_out` / `cost_usd` は書かない — socsim-llm の
/// `MetadataCollector` はトークン数も費用も持たないので，予約名に入れる値が無い．
pub fn log_llm_usage(run: &mut Run, result: &SimulationResult) {
    let total = result.metadata.total();
    let mut values: Vec<(&str, f64)> = vec![
        ("llm_calls", total as f64),
        ("llm_cache_hits", result.metadata.cache_hits() as f64),
    ];
    if total > 0 {
        values.push(("llm_cache_hit_rate", result.metadata.cache_hit_rate()));
    }
    run.log_metrics(SCOPE, &values)
        .expect("LLM 呼び出しの内訳の記録に失敗");
}

// ---------------------------------------------------------------------------
// 試行 (sweep / reproduce の events.jsonl)
// ---------------------------------------------------------------------------

/// `events.jsonl` に書く試行の観測行．
///
/// 予約キーだけを持つ．数はここには書かない — 試行の最終値は下の
/// [`TrialTerminal`] が正本なので，同じ数を 2 箇所に置かない．この行が持つのは
/// 「その試行をいつ見たか」という時間軸だけである．
///
/// `runvault verify --deep` は terminal の `unit_id` が observation にも現れ，
/// かつ両者の `t` が一致することを要求するので，観測した時刻を明示的に残す．
#[derive(Serialize)]
struct TrialObservation<'a> {
    unit_id: &'a str,
    t: u64,
    t_unit: &'static str,
}

/// `events.jsonl` に書く試行の終端行 (旧 `sweep_summary.csv` の 1 行)．
///
/// 先頭 6 フィールドは runvault の予約語．残りは自由欄で，$\psi$ 更新式の重みは
/// 子 run の parameters が持つので繰り返さない．
///
/// シードの欄を `seed` としているのに対し，子 run の parameters 側は `base_seed`
/// という別の名前にしてある．`sweep_events_table` は parameters の列を event の
/// 同名列に上書きするので，同じ名前にすると試行ごとのシードが base seed で
/// 黙って潰れる．
#[derive(Serialize)]
struct TrialTerminal<'a> {
    unit_id: &'a str,
    t: u64,
    t_unit: &'static str,
    outcome: &'static str,
    censored: bool,
    budget: u64,
    seed: u64,
    icc_psi: f64,
    icc_learning_stable: f64,
    beta_psi_l: f64,
    r2_psi_l: f64,
    r2_l_pi: f64,
    beta_l_pi: f64,
    beta_psi_residual: f64,
    beta_psi_residual_p: f64,
    mediation_ratio: f64,
    beta_support_psi: f64,
    beta_support_psi_p: f64,
    efficacy_partial_t: f64,
    hypotheses_supported: u8,
}

/// 試行 1 本を `observation` + `terminal` として書く．
///
/// `sweep` / `reproduce` が見るのは各試行の最終状態だけなので，観測時刻も
/// そこ 1 点である．モデルは収束で止まらず必ず `t_max` まで回るので `outcome` は
/// `horizon`，`censored` は真になる．
pub fn log_trial(
    run: &mut Run,
    index: usize,
    seed: u64,
    final_round: u64,
    t_max: u64,
    rep: &AnchorReport,
) {
    let unit_id = format!("trial-{index}");
    let censored = final_round == t_max;
    run.log_event(
        "observation",
        &TrialObservation {
            unit_id: &unit_id,
            t: final_round,
            t_unit: T_UNIT,
        },
    )
    .unwrap_or_else(|e| panic!("{unit_id} の observation の記録に失敗: {e}"));

    run.log_event(
        "terminal",
        &TrialTerminal {
            unit_id: &unit_id,
            t: final_round,
            t_unit: T_UNIT,
            outcome: if censored { "horizon" } else { "stopped" },
            censored,
            budget: t_max,
            seed,
            icc_psi: rep.icc_psi,
            icc_learning_stable: rep.icc_learning,
            beta_psi_l: rep.beta_psi_l,
            r2_psi_l: rep.r2_psi_l,
            r2_l_pi: rep.r2_l_pi,
            beta_l_pi: rep.beta_l_pi,
            beta_psi_residual: rep.beta_psi_residual,
            beta_psi_residual_p: rep.beta_psi_residual_p,
            mediation_ratio: rep.mediation_ratio,
            beta_support_psi: rep.beta_support_psi,
            beta_support_psi_p: rep.beta_support_psi_p,
            efficacy_partial_t: rep.efficacy_partial_t,
            hypotheses_supported: rep.hypotheses_supported,
        },
    )
    .unwrap_or_else(|e| panic!("{unit_id} の terminal イベントの記録に失敗: {e}"));
}

/// 試行群の平均．[`AnchorReport`] の 13 項目を項目ごとに平均する．
pub fn mean_report(reps: &[AnchorReport]) -> AnchorReport {
    assert!(!reps.is_empty(), "試行が 1 本もありません");
    let n = reps.len() as f64;
    let mean = |f: &dyn Fn(&AnchorReport) -> f64| reps.iter().map(f).sum::<f64>() / n;
    AnchorReport {
        icc_psi: mean(&|r| r.icc_psi),
        icc_learning: mean(&|r| r.icc_learning),
        beta_psi_l: mean(&|r| r.beta_psi_l),
        r2_psi_l: mean(&|r| r.r2_psi_l),
        r2_l_pi: mean(&|r| r.r2_l_pi),
        beta_l_pi: mean(&|r| r.beta_l_pi),
        beta_psi_residual: mean(&|r| r.beta_psi_residual),
        beta_psi_residual_p: mean(&|r| r.beta_psi_residual_p),
        mediation_ratio: mean(&|r| r.mediation_ratio),
        beta_support_psi: mean(&|r| r.beta_support_psi),
        beta_support_psi_p: mean(&|r| r.beta_support_psi_p),
        efficacy_partial_t: mean(&|r| r.efficacy_partial_t),
        hypotheses_supported: (mean(&|r| r.hypotheses_supported as f64)).round() as u8,
    }
}

/// `sweep` のセル 1 つを 1 つの値で表す指標．
///
/// 試行ごとの値は `events.jsonl` の担当なので，ここには集約しか書かない．試行
/// ごとの `icc_psi` を指標にすると (`run_uid`, `step`, `scope`, `name`) が重複
/// する．散らばりが要る図は `events.jsonl` から組み直す．
pub fn log_cell_summary(run: &mut Run, reps: &[AnchorReport]) {
    let m = mean_report(reps);
    run.log_metrics(
        SCOPE,
        &[
            ("n_units", reps.len() as f64),
            ("mean_icc_psi", m.icc_psi),
            ("mean_beta_psi_l", m.beta_psi_l),
            ("mean_r2_psi_l", m.r2_psi_l),
            ("mean_r2_l_pi", m.r2_l_pi),
            ("mean_mediation_ratio", m.mediation_ratio),
            ("mean_beta_support_psi", m.beta_support_psi),
            ("mean_efficacy_partial_t", m.efficacy_partial_t),
            ("mean_hypotheses_supported", m.hypotheses_supported as f64),
        ],
    )
    .expect("セル集約の記録に失敗");
}

/// `reproduce` の run 全体を 1 つの値で表す指標．
///
/// 名前は原著の報告値と同じにしてある — `reference.csv` に入るのは «原著が印字
/// した数» で，同じ名前の run スコープ指標が «この再現が出した数» になり，両者の
/// 差が後から計算できる．`sweep` のセル集約が `mean_*` を名乗るのとは目的が違う
/// (あちらはセル内の散らばりを潰した集約であって，論文の数と並べる量ではない)．
pub fn log_reproduce_summary(run: &mut Run, reps: &[AnchorReport]) {
    let m = mean_report(reps);
    run.log_metrics(
        SCOPE,
        &[
            ("n_units", reps.len() as f64),
            ("icc_psi", m.icc_psi),
            ("icc_learning_stable", m.icc_learning),
            ("beta_psi_l", m.beta_psi_l),
            ("r2_psi_l", m.r2_psi_l),
            ("r2_l_pi", m.r2_l_pi),
            ("beta_l_pi", m.beta_l_pi),
            ("beta_psi_residual", m.beta_psi_residual),
            ("beta_psi_residual_p", m.beta_psi_residual_p),
            ("mediation_ratio", m.mediation_ratio),
            ("beta_support_psi", m.beta_support_psi),
            ("beta_support_psi_p", m.beta_support_psi_p),
            ("efficacy_partial_t", m.efficacy_partial_t),
            ("hypotheses_supported", m.hypotheses_supported as f64),
        ],
    )
    .expect("reproduce の集約の記録に失敗");
}

// ---------------------------------------------------------------------------
// シードの派生
// ---------------------------------------------------------------------------

/// 反復 1 本のシードを base seed から決定的に派生させる (`run` / `reproduce`)．
pub fn replicate_seed(base: u64, index: usize) -> u64 {
    socsim_core::derive_seed(base, &[index as u64])
}

/// 試行 1 本のシードを base seed とセル座標から決定的に派生させる (`sweep`)．
pub fn trial_seed(base: u64, alpha: f64, delta: f64, index: usize) -> u64 {
    socsim_core::derive_seed(
        base,
        &[
            (alpha * 1000.0) as u64,
            (delta * 1000.0) as u64,
            index as u64,
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_inputs_give_the_same_seed() {
        assert_eq!(replicate_seed(1999, 3), replicate_seed(1999, 3));
        assert_eq!(
            trial_seed(1999, 0.3, 0.35, 2),
            trial_seed(1999, 0.3, 0.35, 2)
        );
    }

    #[test]
    fn each_coordinate_changes_the_seed() {
        let base = trial_seed(1999, 0.3, 0.35, 0);
        assert_ne!(base, trial_seed(2000, 0.3, 0.35, 0), "base が効いていない");
        assert_ne!(base, trial_seed(1999, 0.4, 0.35, 0), "alpha が効いていない");
        assert_ne!(base, trial_seed(1999, 0.3, 0.45, 0), "delta が効いていない");
        assert_ne!(base, trial_seed(1999, 0.3, 0.35, 1), "index が効いていない");
    }

    /// 具体値を固定する．
    ///
    /// ここが変わるのは socsim の `derive_seed` が変わったときで，そのときは
    /// 過去の run と結果を比較できなくなっている．`Cargo.lock` が socsim の
    /// commit を固定しているので，この値は依存を上げたときにだけ動く．
    #[test]
    fn golden_values_are_pinned() {
        assert_eq!(replicate_seed(1999, 0), 13_926_578_433_687_357_256);
        // 旧 `sweep_summary.csv` の (α=.1, δ=.2, run=0) 行の seed 列そのもの．
        assert_eq!(trial_seed(1999, 0.1, 0.2, 0), 4_204_546_973_476_058_460);
    }

    #[test]
    fn the_mean_report_averages_every_field() {
        let a = AnchorReport {
            icc_psi: 0.2,
            icc_learning: 0.1,
            beta_psi_l: 1.0,
            r2_psi_l: 0.4,
            r2_l_pi: 0.2,
            beta_l_pi: 0.5,
            beta_psi_residual: 0.1,
            beta_psi_residual_p: 0.3,
            mediation_ratio: 0.6,
            beta_support_psi: 0.4,
            beta_support_psi_p: 0.01,
            efficacy_partial_t: -1.0,
            hypotheses_supported: 5,
        };
        let b = AnchorReport {
            icc_psi: 0.4,
            hypotheses_supported: 7,
            ..a.clone()
        };
        let m = mean_report(&[a, b]);
        assert!((m.icc_psi - 0.3).abs() < 1e-15);
        assert_eq!(m.hypotheses_supported, 6);
    }
}
