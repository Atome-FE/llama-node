use std::str::FromStr;
use tokenizers::tokenizer::{Encoding as HfEncoding, Tokenizer as HfTokenizer};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Tokenizer {
    tokenizer: HfTokenizer,
}

#[wasm_bindgen]
impl Tokenizer {
    #[wasm_bindgen(constructor)]
    pub fn from_string(json: String) -> Tokenizer {
        Tokenizer {
            tokenizer: HfTokenizer::from_str(json.as_str()).unwrap(),
        }
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Encoding {
        Encoding {
            encoding: self.tokenizer.encode(text, add_special_tokens).unwrap(),
        }
    }

    pub fn decode(&self, ids: js_sys::Uint32Array, add_special_tokens: bool) -> String {
        let ids: Vec<u32> = ids.to_vec();
        self.tokenizer.decode(ids, add_special_tokens).unwrap()
    }
}

#[wasm_bindgen]
pub struct Encoding {
    encoding: HfEncoding,
}

#[wasm_bindgen]
impl Encoding {
    #[wasm_bindgen(method, getter = ids)]
    pub fn get_ids(&self) -> js_sys::Uint32Array {
        self.encoding.get_ids().into()
    }

    #[wasm_bindgen(method, getter = tokens)]
    pub fn get_tokens(&self) -> js_sys::Array {
        self.encoding
            .get_tokens()
            .iter()
            .map(|x| js_sys::JsString::from(x.as_str()))
            .collect()
    }
}
