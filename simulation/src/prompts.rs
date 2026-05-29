//! LLM prompt construction and response parsing for `voice_decision`.
//!
//! The individual is shown their inner state (ψ, fear, ivt, private concern) and
//! their team context (support, coaching, shared belief ψ̄) and asked whether to
//! VOICE a concern or stay SILENT, and — if voicing — which learning behavior
//! they engage in. The LLM returns a JSON object of the form
//!
//! ```json
//! { "decision": "voice" | "silence", "behavior": "speak" | "help" | "error_talk" | null }
//! ```
//!
//! Parse failures fall back to `Silence` (`parse_failed = true`).

use serde::Deserialize;
use serde_json::Value;

use crate::world::{Individual, TeamWorld};

// --------------------------------------------------------------------------- //
// Learning behavior kind
// --------------------------------------------------------------------------- //

/// Which learning behavior an individual engaged in this step (when voicing).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Behavior {
    /// Speaking up / feedback-seeking (voice).
    Speak,
    /// Asking for help.
    Help,
    /// Talking about an error openly.
    ErrorTalk,
}

impl Behavior {
    /// Stable label.
    pub fn label(&self) -> &'static str {
        match self {
            Behavior::Speak => "speak",
            Behavior::Help => "help",
            Behavior::ErrorTalk => "error_talk",
        }
    }

    /// Parse a behavior label.
    pub fn parse(s: &str) -> Option<Behavior> {
        match s.trim().to_ascii_lowercase().as_str() {
            "speak" | "voice" | "feedback" => Some(Behavior::Speak),
            "help" | "ask_help" | "help_seeking" => Some(Behavior::Help),
            "error_talk" | "error" | "discuss_error" => Some(Behavior::ErrorTalk),
            _ => None,
        }
    }
}

// --------------------------------------------------------------------------- //
// Prompt construction
// --------------------------------------------------------------------------- //

/// Build the voice-decision prompt for `agent_id` from the world.
pub fn build_voice_prompt(world: &TeamWorld, agent_id: socsim_core::AgentId) -> String {
    let ind = &world.individuals[&agent_id];
    let team = &world.teams[&ind.team];
    let context = format_context(ind, team.support, team.coaching, team.psi_bar);

    format!(
        "You are an employee on a work team. An operational problem or possible mistake \
         has surfaced. You must decide whether to SPEAK UP / seek help / discuss the error \
         openly (VOICE) or to stay SILENT.\n\n\
         Your inner state and team context:\n\
         {context}\n\
         Reply with a SINGLE JSON object on one line:\n\
         {{\"decision\": \"voice\" | \"silence\", \
            \"behavior\": \"speak\" | \"help\" | \"error_talk\" | null}}\n\
         Rules: if decision = voice, behavior must be one of the three labels (the learning \
         behavior you engaged in). If decision = silence, behavior must be null. Output JSON only."
    )
}

fn format_context(ind: &Individual, support: f64, coaching: f64, psi_bar: f64) -> String {
    format!(
        "  psychological safety   ψ = {psi:.2}\n\
         \x20 fear of consequences   f = {fear:.2}\n\
         \x20 implicit-voice norms   θ = {ivt:.2}\n\
         \x20 private concern        c = {concern:.2}\n\
         \x20 team context support   s = {support:.2}\n\
         \x20 leader coaching        k = {coaching:.2}\n\
         \x20 team shared belief     ψ̄ = {psi_bar:.2}\n",
        psi = ind.psi,
        fear = ind.fear,
        ivt = ind.ivt,
        concern = ind.private_concern,
        support = support,
        coaching = coaching,
        psi_bar = psi_bar,
    )
}

// --------------------------------------------------------------------------- //
// Response parsing
// --------------------------------------------------------------------------- //

/// Parsed voice-decision verdict.
#[derive(Debug, Clone, PartialEq)]
pub struct VoiceVerdict {
    /// True iff the individual voiced.
    pub voiced: bool,
    /// The learning behavior when voicing; `None` on silence.
    pub behavior: Option<Behavior>,
    /// True if parsing failed and we fell back to `Silence`.
    pub parse_failed: bool,
}

#[derive(Deserialize)]
struct RawDecision {
    decision: Option<String>,
    behavior: Option<String>,
}

/// Parse an LLM response into a verdict.
///
/// Lenient: extracts the first balanced `{...}` object, accepts mixed-case
/// labels, and on any failure falls back to `Silence`.
pub fn parse_voice_decision(text: &str) -> VoiceVerdict {
    let fallback = VoiceVerdict {
        voiced: false,
        behavior: None,
        parse_failed: true,
    };

    let json_str = match extract_json_object(text) {
        Some(s) => s,
        None => return fallback,
    };

    if let Ok(raw) = serde_json::from_str::<RawDecision>(&json_str) {
        return finalise_verdict(raw);
    }
    if let Ok(val) = serde_json::from_str::<Value>(&json_str) {
        let raw = RawDecision {
            decision: val
                .get("decision")
                .and_then(|v| v.as_str().map(str::to_string)),
            behavior: val.get("behavior").and_then(|v| {
                if v.is_null() {
                    None
                } else {
                    v.as_str().map(str::to_string)
                }
            }),
        };
        return finalise_verdict(raw);
    }
    fallback
}

fn finalise_verdict(raw: RawDecision) -> VoiceVerdict {
    let decision = raw
        .decision
        .as_deref()
        .map(|s| s.trim().to_ascii_lowercase())
        .unwrap_or_default();
    match decision.as_str() {
        "voice" | "speak" | "speak_up" => {
            // Behavior defaults to Speak if missing / unparseable.
            let behavior = raw
                .behavior
                .as_deref()
                .and_then(Behavior::parse)
                .unwrap_or(Behavior::Speak);
            VoiceVerdict {
                voiced: true,
                behavior: Some(behavior),
                parse_failed: false,
            }
        }
        "silence" | "silent" | "withhold" => VoiceVerdict {
            voiced: false,
            behavior: None,
            parse_failed: false,
        },
        _ => VoiceVerdict {
            voiced: false,
            behavior: None,
            parse_failed: true,
        },
    }
}

/// Extract the first balanced `{...}` substring from `text`.
fn extract_json_object(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let start = bytes.iter().position(|&b| b == b'{')?;
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_str {
            if esc {
                esc = false;
            } else if b == b'\\' {
                esc = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(text[start..=i].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_canonical_voice() {
        let v = parse_voice_decision(r#"{"decision": "voice", "behavior": "error_talk"}"#);
        assert!(v.voiced);
        assert_eq!(v.behavior, Some(Behavior::ErrorTalk));
        assert!(!v.parse_failed);
    }

    #[test]
    fn parses_silence() {
        let v = parse_voice_decision(r#"{"decision":"silence","behavior":null}"#);
        assert!(!v.voiced);
        assert_eq!(v.behavior, None);
        assert!(!v.parse_failed);
    }

    #[test]
    fn voice_without_behavior_defaults_to_speak() {
        let v = parse_voice_decision(r#"{"decision":"voice"}"#);
        assert!(v.voiced);
        assert_eq!(v.behavior, Some(Behavior::Speak));
    }

    #[test]
    fn tolerates_surrounding_text() {
        let v = parse_voice_decision(r#"Sure: {"decision":"voice","behavior":"help"}. Done."#);
        assert!(v.voiced);
        assert_eq!(v.behavior, Some(Behavior::Help));
    }

    #[test]
    fn unknown_decision_falls_back() {
        let v = parse_voice_decision(r#"{"decision":"???"}"#);
        assert!(v.parse_failed);
        assert!(!v.voiced);
    }

    #[test]
    fn no_json_falls_back() {
        let v = parse_voice_decision("no json here");
        assert!(v.parse_failed);
        assert!(!v.voiced);
    }
}
