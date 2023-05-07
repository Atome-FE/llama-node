all: nodejs web

nodejs:
	wasm-pack build --target nodejs --out-dir ./nodejs --out-name tokenizer-node --scope llama-node --release --weak-refs

web:
	wasm-pack build --target web --out-dir ./web --out-name tokenizer-web --scope llama-node --release --weak-refs

clean:
	rm -rf ./nodejs
	rm -rf ./web