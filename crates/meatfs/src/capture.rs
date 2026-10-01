//! Content capture for recorded diagnostics.
//!
//! Operational state (paths, bindings, live grant targets) keeps the names
//! it needs to act. Records — journal events, receipts, error records —
//! retain source-controlled text only as the host's capture policy allows,
//! decided when the record is made. A record made under `Omit` holds no copy
//! of the text, so no later view or policy change can reveal it.

use crate::{Path, Value};

/// What a record may retain of content: payloads and source-controlled text
/// such as names and paths. Chosen by the host; defaults to `Omit`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ContentCapture {
    /// Keep identities, structure and statuses; retain no content.
    #[default]
    Omit,
    /// Retain content inline.
    Inline,
}

/// Content as a record retains it.
#[derive(Debug, Clone, PartialEq)]
pub enum CapturedValue {
    Omitted,
    Inline(Value),
}

impl ContentCapture {
    /// Record a value. Under `Omit` nothing is copied.
    pub fn value(self, value: &Value) -> CapturedValue {
        match self {
            ContentCapture::Omit => CapturedValue::Omitted,
            ContentCapture::Inline => CapturedValue::Inline(value.clone()),
        }
    }

    /// Record a name or path as text. Under `Omit` nothing is copied.
    pub fn name(self, path: &Path) -> CapturedValue {
        match self {
            ContentCapture::Omit => CapturedValue::Omitted,
            ContentCapture::Inline => CapturedValue::Inline(Value::Text(path.to_string())),
        }
    }

    /// Record a source-controlled label, such as an output name.
    pub fn label(self, label: &str) -> CapturedValue {
        match self {
            ContentCapture::Omit => CapturedValue::Omitted,
            ContentCapture::Inline => CapturedValue::Inline(Value::from(label)),
        }
    }

    /// Record free text, built only if it will be retained.
    pub fn text(self, text: impl FnOnce() -> String) -> Option<String> {
        match self {
            ContentCapture::Omit => None,
            ContentCapture::Inline => Some(text()),
        }
    }
}

impl CapturedValue {
    /// This record as seen through `view`: an `Omit` view never shows
    /// content, whatever the record retained; an `Inline` view shows what
    /// was retained and cannot restore what was not.
    pub fn viewed(&self, view: ContentCapture) -> CapturedValue {
        match view {
            ContentCapture::Omit => CapturedValue::Omitted,
            ContentCapture::Inline => self.clone(),
        }
    }
}
