//! `--dump-tokens` / `--dump-logprobs` writers.
//!
//! The JSON schema matches ds4's official-API capture format used by
//! the F005 acceptance vectors (`tests/dsv4-vectors/`): one
//! [`LogprobStep`] per decode step, top-K entries sorted by
//! descending probability.
//!
//! The file is written incrementally so a long run is still
//! debuggable if interrupted — we keep one open writer for the
//! lifetime of the dump and emit one JSON line per step
//! (newline-delimited JSON, "JSONL"). Callers that need a single
//! JSON array can post-process with `jq -s`.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use serde::Serialize;

use crate::CliError;

/// One top-K logprob entry. `prob` is the post-filter renormalized
/// probability (so it's directly comparable with the API capture's
/// `linear_probability` field). `logit` is the pre-softmax raw value
/// for callers that want to diff against pre-temp model output.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct LogprobEntry {
    pub token_id: u32,
    pub logit: f32,
    pub prob: f32,
}

/// One decode-step record. `chosen_id` is the actually sampled token;
/// `top_k` is the top-K candidate list at this step (already sorted
/// by descending probability).
#[derive(Debug, Clone, Serialize)]
pub struct LogprobStep {
    pub step: u32,
    pub chosen_id: u32,
    pub top_k: Vec<LogprobEntry>,
}

/// JSONL writer for `--dump-logprobs`. Wraps a `BufWriter<File>` and
/// flushes per line so output is durable across crashes.
pub struct LogprobDumper {
    writer: BufWriter<File>,
    step: u32,
    top_k_cap: usize,
}

impl LogprobDumper {
    /// Create a dumper writing to `path`. `top_k_cap` is the maximum
    /// number of entries per step (the caller normally hands us
    /// fewer — the cap exists to clamp pathological values).
    pub fn create(path: &Path, top_k_cap: usize) -> Result<Self, CliError> {
        let file = File::create(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
            step: 0,
            top_k_cap,
        })
    }

    /// Write one decode-step entry. Logits and probabilities are
    /// caller-supplied — the dumper does no sorting / filtering.
    pub fn write_step(&mut self, chosen_id: u32, top: &[LogprobEntry]) -> Result<(), CliError> {
        let truncated: Vec<LogprobEntry> = top.iter().take(self.top_k_cap).copied().collect();
        let rec = LogprobStep {
            step: self.step,
            chosen_id,
            top_k: truncated,
        };
        // `serde_json::to_writer` doesn't add a trailing newline.
        serde_json::to_writer(&mut self.writer, &rec).map_err(|e| {
            CliError::Io(std::io::Error::other(format!(
                "dump-logprobs serialize failed: {e}"
            )))
        })?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()?;
        self.step += 1;
        Ok(())
    }
}

/// `--dump-tokens` writer — writes one line per token to `stderr`.
/// Format: `step=NN id=NNNN text="..."`.
pub fn emit_token_line(step: u32, token_id: u32, text: &str) {
    // Use eprintln so it lands on stderr; trace output goes to stdout
    // by default and we want the token stream interleaved cleanly.
    eprintln!("step={step} id={token_id} text={text:?}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn write_step_appends_jsonl_line() {
        let tmp = std::env::temp_dir().join("rsllm-dump-test.jsonl");
        let _ = std::fs::remove_file(&tmp);
        {
            let mut d = LogprobDumper::create(&tmp, 5).unwrap();
            d.write_step(
                42,
                &[
                    LogprobEntry {
                        token_id: 42,
                        logit: 5.0,
                        prob: 0.7,
                    },
                    LogprobEntry {
                        token_id: 99,
                        logit: 3.0,
                        prob: 0.2,
                    },
                ],
            )
            .unwrap();
            d.write_step(7, &[]).unwrap();
        }
        let mut s = String::new();
        File::open(&tmp).unwrap().read_to_string(&mut s).unwrap();
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("\"chosen_id\":42"));
        assert!(lines[0].contains("\"step\":0"));
        assert!(lines[1].contains("\"chosen_id\":7"));
        assert!(lines[1].contains("\"step\":1"));
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn write_step_truncates_to_cap() {
        let tmp = std::env::temp_dir().join("rsllm-dump-cap-test.jsonl");
        let _ = std::fs::remove_file(&tmp);
        {
            let mut d = LogprobDumper::create(&tmp, 2).unwrap();
            let entries: Vec<LogprobEntry> = (0..10)
                .map(|i| LogprobEntry {
                    token_id: i,
                    logit: i as f32,
                    prob: 0.1,
                })
                .collect();
            d.write_step(0, &entries).unwrap();
        }
        let s = std::fs::read_to_string(&tmp).unwrap();
        // Two entries → token_ids 0 and 1 only.
        assert!(s.contains("\"token_id\":0"));
        assert!(s.contains("\"token_id\":1"));
        assert!(!s.contains("\"token_id\":2"));
        let _ = std::fs::remove_file(&tmp);
    }
}
