LLM Pipeline Explorer

LLM Pipeline Explorer is a small, interactive teaching app that helps explain how text is prepared and represented in a language model pipeline. Build a vocabulary from sample sentences, tokenize text into visible pieces, and explore sentence embeddings in an interactive 3D chart.

## Features

- Build a word vocabulary with token IDs and frequency counts.
- Tokenize sentences into colored chips and see how unknown words are split into smaller pieces.
- Create spelling-based embeddings from character groups and compare sentences by cosine similarity.
- Optionally explore meaning-based embeddings with Transformers.js and the `Xenova/all-MiniLM-L6-v2` model.
- Add your own text, tokenize it, and compare it with the plotted sentences.

## Built with

- HTML, CSS, and plain JavaScript
- Plotly.js for the 3D chart
- Transformers.js for optional meaning-based embeddings

## Run locally

The spelling-based mode can run in a modern browser. Meaning-based mode needs the page served over HTTP because the model loads from a CDN.

1. Download or clone this repository.
2. Open a terminal in the project folder.
3. Start a local static server. For example, if Python is installed, run:

   ```sh
   python3 -m http.server 8000
   ```

4. Open [http://localhost:8000](http://localhost:8000) in your browser.
5. Choose a mode and click **Embed and plot**. The first use of meaning-based mode may take time to download the model.

No npm install, backend, or API key is required.

## How to use

1. Enter training sentences and click **Build vocabulary**.
2. Enter a sentence and click **Tokenize** to inspect its tokens.
3. Review or edit the sample sentences, choose an embedding mode, and click **Embed and plot**.
4. Enter your own word or sentence and click **Tokenize, embed & add to plot** to add it to the visualization.

## Note

Meaning-based embeddings require an internet connection to load Transformers.js and the model from their CDNs. If loading fails, the app falls back to spelling-based embeddings.
