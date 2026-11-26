    // WHAT THIS DOES: Stores the active vocabulary, all sorted training words, and whether token results are currently displayed
    let vocabulary = [];
    let sortedVocabularyTokens = [];
    let vocabularyBuilt = false;
    let hasTokenizedSentence = false;
    let lastTokenizedSentence = '';
    let meaningModelPromise = null;
    let embeddingInProgress = false;
    let latestEmbeddingData = null;
    let showPointLabels = true;
    let activeChartView = '3d';

    // WHAT THIS DOES: Applies the same lowercase and punctuation spacing rules to training and input text
    function normalizeText(text) {
      return text.toLowerCase()
        .replace(/\./g, ' . ')
        .replace(/,/g, ' , ')
        .replace(/!/g, ' ! ')
        .replace(/\?/g, ' ? ')
        .replace(/'/g, " ' ");
    }

    // WHAT THIS DOES: Processes the training sentences to build a sorted vocabulary with token frequencies and IDs
    function buildVocabulary() {
      const trainingText = document.getElementById('training-sentences').value;

      // Split normalized text into tokens, ignoring empty tokens
      const cleanedText = normalizeText(trainingText);
      const rawTokens = cleanedText.split(/\s+/).filter(token => token.length > 0);

      // Count how often each token appears
      const counts = new Map();
      for (const token of rawTokens) {
        counts.set(token, (counts.get(token) || 0) + 1);
      }

      // Sort tokens from most frequent to least frequent. For ties, sort alphabetically
      const uniqueTokens = Array.from(counts.keys());
      uniqueTokens.sort((a, b) => {
        const countDifference = counts[b] - counts[a];
        if (countDifference !== 0) {
          return countDifference;
        }
        return a.localeCompare(b);
      });

      sortedVocabularyTokens = uniqueTokens.map(token => ({
        token: token,
        count: counts.get(token)
      }));

      const slider = document.getElementById('vocab-limit-slider');
      const total = sortedVocabularyTokens.length;
      slider.min = total === 0 ? '0' : '1';
      slider.max = String(total);
      slider.value = String(total);
      slider.disabled = total === 0;
      vocabularyBuilt = true;
      document.getElementById('vocab-limit-controls').style.display = 'block';

      // Display the built vocabulary in the UI
      renderVocabulary();
      if (hasTokenizedSentence) {
        tokenizeSentence();
      }
    }

    // WHAT THIS DOES: Splits known words into vocabulary tokens and unknown words into character pieces, then displays their IDs and statistics
    function tokenizeSentence(sentence = document.getElementById('encode-sentence').value) {
      const results = document.getElementById('tokenize-results');
      const message = document.getElementById('tokenize-message');
      const details = document.getElementById('tokenize-details');
      hasTokenizedSentence = false;
      results.style.display = 'block';
      message.style.display = 'none';
      details.style.display = 'none';

      if (!vocabularyBuilt) {
        message.textContent = 'Build the vocabulary first by clicking "Build vocabulary".';
        message.style.display = 'block';
        return;
      }

      lastTokenizedSentence = sentence;
      const tokens = normalizeText(sentence).split(/\s+/).filter(token => token.length > 0);
      const vocabularyIds = new Map(vocabulary.map(item => [item.token, item.id]));
      const tokenPieces = [];

      for (const token of tokens) {
        if (vocabularyIds.has(token)) {
          tokenPieces.push({ text: token, id: vocabularyIds.get(token) });
          continue;
        }

        const characters = Array.from(token);
        for (let i = 0; i < characters.length; i++) {
          tokenPieces.push({
            text: i === 0 ? characters[i] : `##${characters[i]}`,
            id: 0
          });
        }
      }

      const characterCount = Array.from(sentence).length;
      document.getElementById('character-count').textContent = characterCount;
      document.getElementById('token-count').textContent = tokenPieces.length;
      document.getElementById('characters-per-token').textContent = tokenPieces.length
        ? (characterCount / tokenPieces.length).toFixed(2)
        : 'N/A (no tokens)';

      const tokenList = document.getElementById('token-list');
      tokenList.innerHTML = '';
      for (const piece of tokenPieces) {
        const chip = document.createElement('span');
        chip.className = 'token-chip';
        chip.style.setProperty('--token-hue', (piece.id * 137.508) % 360);

        const tokenText = document.createElement('span');
        tokenText.textContent = piece.text;
        chip.appendChild(tokenText);

        const tokenId = document.createElement('span');
        tokenId.className = 'token-chip-id';
        tokenId.textContent = `ID: ${piece.id}`;
        chip.appendChild(tokenId);
        tokenList.appendChild(chip);
      }

      details.style.display = 'block';
      hasTokenizedSentence = true;
    }

    // WHAT THIS DOES: Counts hashed character trigrams and normalizes their 64-number vector
    function createSpellingVector(sentence) {
      const characters = Array.from(sentence.toLowerCase())
        .filter(character => /\p{L}/u.test(character));
      const vector = Array(64).fill(0);

      for (let i = 0; i <= characters.length - 3; i++) {
        const trigram = characters.slice(i, i + 3).join('');
        let hash = 0;
        // A polynomial rolling hash multiplies by 31, adds each letter code, then maps to one of 64 buckets.
        for (const character of trigram) {
          hash = (hash * 31 + character.codePointAt(0)) >>> 0;
        }
        vector[hash % 64] += 1;
      }

      const length = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
      return length > 0 ? vector.map(value => value / length) : vector;
    }

    // WHAT THIS DOES: Projects an embedding vector of any size into three chart coordinates
    function projectToThreeDimensions(vector) {
      let x = 0;
      let y = 0;
      let z = 0;

      // Fixed cosine and sine waves assign each vector position a repeatable contribution to x, y, and z.
      for (let i = 0; i < vector.length; i++) {
        const angle = (i / vector.length) * Math.PI * 2;
        x += vector[i] * Math.cos(angle);
        y += vector[i] * Math.sin(angle);
        z += vector[i] * Math.cos(angle * 2);
      }

      return { x, y, z };
    }

    // WHAT THIS DOES: Calculates cosine similarity for any two embedding vectors
    function cosineSimilarity(firstVector, secondVector) {
      let dotProduct = 0;
      let firstLength = 0;
      let secondLength = 0;

      for (let i = 0; i < firstVector.length; i++) {
        dotProduct += firstVector[i] * secondVector[i];
        firstLength += firstVector[i] * firstVector[i];
        secondLength += secondVector[i] * secondVector[i];
      }

      const lengthProduct = Math.sqrt(firstLength) * Math.sqrt(secondLength);
      return lengthProduct === 0 ? 0 : dotProduct / lengthProduct;
    }

    // WHAT THIS DOES: Finds the most similar unique sentence pairs from the current embeddings
    function findClosestPairs(sentences, vectors, limit = 5) {
      const pairs = [];
      for (let first = 0; first < sentences.length; first++) {
        for (let second = first + 1; second < sentences.length; second++) {
          pairs.push({
            firstSentence: sentences[first],
            secondSentence: sentences[second],
            score: cosineSimilarity(vectors[first], vectors[second])
          });
        }
      }

      return pairs.sort((a, b) => b.score - a.score).slice(0, limit);
    }

    // WHAT THIS DOES: Ranks the closest sentences to one selected point, excluding that point itself
    function findClosestSentences(sentences, vectors, targetIndex, limit = 3) {
      return sentences
        .map((sentence, index) => ({
          sentence: sentence,
          index: index,
          score: cosineSimilarity(vectors[targetIndex], vectors[index])
        }))
        .filter(item => item.index !== targetIndex)
        .sort((a, b) => b.score - a.score)
        .slice(0, limit);
    }

    // WHAT THIS DOES: Displays the five highest-scoring sentence pairs or a helpful message
    function renderClosestPairs(sentences, vectors) {
      const tableContainer = document.getElementById('closest-pairs-table-container');
      const tableBody = document.getElementById('closest-pairs-table-body');
      const message = document.getElementById('closest-pairs-message');
      tableBody.innerHTML = '';

      if (sentences.length < 2) {
        tableContainer.style.display = 'none';
        message.textContent = 'At least two sentences are needed to compare similarity.';
        message.style.display = 'block';
        return;
      }

      for (const pair of findClosestPairs(sentences, vectors)) {
        const row = document.createElement('tr');
        for (const value of [
          pair.firstSentence,
          pair.secondSentence,
          pair.score.toFixed(3)
        ]) {
          const cell = document.createElement('td');
          cell.textContent = value;
          row.appendChild(cell);
        }
        tableBody.appendChild(row);
      }

      message.style.display = 'none';
      tableContainer.style.display = 'block';
    }

    // WHAT THIS DOES: Updates the dashboard metrics to describe the latest successful embedding run
    function renderEmbeddingSummary(sentences, vectors, mode, usedFallback) {
      const topPair = findClosestPairs(sentences, vectors, 1)[0];
      document.getElementById('summary-sentences').textContent = sentences.length;
      document.getElementById('summary-dimensions').textContent = vectors[0].length;
      document.getElementById('summary-method').textContent = mode === 'meaning'
        ? 'Meaning · MiniLM'
        : usedFallback ? 'Spelling fallback' : 'Spelling · 3-grams';
      document.getElementById('summary-similarity').textContent = topPair
        ? topPair.score.toFixed(3)
        : 'N/A';
      document.getElementById('embedding-summary').style.display = 'grid';
    }

    // WHAT THIS DOES: Builds either a 3D projection or a cosine-similarity heatmap from the active vectors
    function createChartFigure() {
      const { sentences, vectors, coordinates, highlightedIndex } = latestEmbeddingData;

      if (activeChartView === 'similarity') {
        const labels = sentences.map((sentence, index) => {
          const shortSentence = sentence.length > 22
            ? `${sentence.slice(0, 19)}...`
            : sentence;
          return `S${index + 1}: ${shortSentence}`;
        });
        const similarities = vectors.map(firstVector => (
          vectors.map(secondVector => cosineSimilarity(firstVector, secondVector))
        ));
        const sentencePairs = sentences.map(firstSentence => (
          sentences.map(secondSentence => [firstSentence, secondSentence])
        ));

        return {
          data: [{
            type: 'heatmap',
            x: labels,
            y: labels,
            z: similarities,
            customdata: sentencePairs,
            colorscale: [
              [0, '#c2410c'],
              [0.5, '#f8fafc'],
              [1, '#4338ca']
            ],
            zmin: -1,
            zmax: 1,
            colorbar: {
              title: 'Cosine similarity',
              tickformat: '.1f'
            },
            hovertemplate: '%{customdata[0]}<br>vs %{customdata[1]}<br>Cosine similarity: %{z:.3f}<extra></extra>'
          }],
          layout: {
            margin: { l: 150, r: 30, t: 20, b: 130 },
            xaxis: { title: 'Sentence', tickangle: -35, automargin: true },
            yaxis: { title: 'Sentence', autorange: 'reversed', automargin: true }
          }
        };
      }

      const labels = sentences.map(sentence => (
        sentence.length > 30 ? `${sentence.slice(0, 27)}...` : sentence
      ));
      const pointSizes = sentences.map((_, index) => index === highlightedIndex ? 12 : 6);
      const pointColors = sentences.map((_, index) => (
        index === highlightedIndex ? '#f97316' : '#2563eb'
      ));

      return {
        data: [{
          type: 'scatter3d',
          mode: showPointLabels ? 'markers+text' : 'markers',
          x: coordinates.map(point => point.x),
          y: coordinates.map(point => point.y),
          z: coordinates.map(point => point.z),
          text: labels,
          textposition: 'top center',
          customdata: sentences,
          hovertemplate: '%{customdata}<extra></extra>',
          marker: { size: pointSizes, color: pointColors }
        }],
        layout: {
          margin: { l: 0, r: 0, t: 20, b: 0 },
          showlegend: false,
          scene: {
            xaxis: { title: 'X' },
            yaxis: { title: 'Y' },
            zaxis: { title: 'Z' }
          }
        }
      };
    }

    // WHAT THIS DOES: Redraws the active chart view without recalculating any sentence embeddings
    async function renderActiveChart() {
      const chart = document.getElementById('embedding-chart');
      const figure = createChartFigure();
      const renderPlot = chart.data && chart.data.length ? Plotly.react : Plotly.newPlot;
      await renderPlot(chart, figure.data, figure.layout, {
        responsive: true,
        displayModeBar: false
      });
    }

    // WHAT THIS DOES: Switches chart views while keeping the current embedding results
    async function setChartView(view) {
      activeChartView = view;
      const isThreeDimensional = view === '3d';
      document.getElementById('chart-view-3d-btn').setAttribute('aria-pressed', String(isThreeDimensional));
      document.getElementById('chart-view-similarity-btn').setAttribute('aria-pressed', String(!isThreeDimensional));
      document.getElementById('chart-label-toggle').disabled = !isThreeDimensional;
      document.getElementById('chart-view-description').textContent = isThreeDimensional
        ? 'A simple projection combines vector values with cosine and sine waves to create three coordinates. Rotate or zoom the chart to explore sentence positions.'
        : 'Each cell shows cosine similarity between two full sentence vectors; darker purple means more similar.';
      document.getElementById('chart-view-title').textContent = isThreeDimensional
        ? '3D projection'
        : 'Cosine similarity matrix';

      if (!latestEmbeddingData) {
        return;
      }

      try {
        await renderActiveChart();
      } catch (error) {
        const status = document.getElementById('embedding-status');
        status.textContent = 'The chart view could not be updated. Check the browser console and try again.';
        status.classList.add('error');
        status.style.display = 'block';
        console.error('Could not update the chart view.', error);
      }
    }

    // WHAT THIS DOES: Exports the full active vectors and their 3D chart coordinates as a downloadable CSV
    function exportEmbeddingCsv() {
      if (!latestEmbeddingData) {
        return;
      }

      const { sentences, vectors, coordinates, mode } = latestEmbeddingData;
      const dimensions = vectors[0].length;
      const headers = [
        'sentence',
        'embedding_mode',
        'vector_dimensions',
        'x',
        'y',
        'z',
        ...Array.from({ length: dimensions }, (_, index) => `vector_${index + 1}`)
      ];
      const rows = vectors.map((vector, index) => [
        sentences[index],
        mode,
        dimensions,
        coordinates[index].x,
        coordinates[index].y,
        coordinates[index].z,
        ...vector
      ]);
      const csv = [headers, ...rows]
        .map(row => row.map(value => `"${String(value).replace(/"/g, '""')}"`).join(','))
        .join('\r\n');
      const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
      const downloadLink = document.createElement('a');
      downloadLink.href = url;
      downloadLink.download = 'llm-pipeline-embeddings.csv';
      document.body.appendChild(downloadLink);
      downloadLink.click();
      downloadLink.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    }

    // WHAT THIS DOES: Shows or hides point labels without recomputing any sentence embeddings
    async function updatePointLabels() {
      showPointLabels = document.getElementById('chart-label-toggle').checked;
      const chart = document.getElementById('embedding-chart');
      if (activeChartView !== '3d' || !chart.data || typeof Plotly === 'undefined') {
        return;
      }

      try {
        await Plotly.restyle(chart, {
          mode: showPointLabels ? 'markers+text' : 'markers'
        });
      } catch (error) {
        const status = document.getElementById('embedding-status');
        status.textContent = 'Could not update chart labels. Try embedding again.';
        status.classList.add('error');
        status.style.display = 'block';
        throw error;
      }
    }

    // WHAT THIS DOES: Shows the custom sentence's first embedding values and its three closest neighbors
    function renderCustomSentenceResults(sentences, vectors, targetIndex) {
      document.getElementById('custom-embedding-values').textContent = vectors[targetIndex]
        .slice(0, 8)
        .map(value => value.toFixed(3))
        .join(', ');

      const neighbors = document.getElementById('custom-neighbors');
      neighbors.innerHTML = '';
      for (const neighbor of findClosestSentences(sentences, vectors, targetIndex)) {
        const item = document.createElement('li');
        item.textContent = `${neighbor.sentence} — ${neighbor.score.toFixed(3)}`;
        neighbors.appendChild(item);
      }
      document.getElementById('custom-sentence-results').style.display = 'block';
    }

    // WHAT THIS DOES: Loads the feature-extraction library and model once while reporting download progress
    async function loadMeaningModel(status) {
      if (!meaningModelPromise) {
        meaningModelPromise = (async () => {
          const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2');
          return pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
            progress_callback: progress => {
              const file = progress.file || progress.status || 'model files';
              const percent = typeof progress.progress === 'number'
                ? ` (${progress.progress.toFixed(0)}%)`
                : '';
              status.textContent = `Loading meaning model: ${file}${percent}`;
            }
          });
        })();
      }

      try {
        return await meaningModelPromise;
      } catch (error) {
        meaningModelPromise = null;
        throw error;
      }
    }

    // WHAT THIS DOES: Prevents overlapping chart updates while the active embedding mode is running
    async function embedAndPlotSentences(highlightedIndex = null) {
      if (embeddingInProgress) {
        return;
      }

      embeddingInProgress = true;
      const controls = [
        document.getElementById('embed-plot-btn'),
        document.getElementById('add-custom-sentence-btn'),
        document.getElementById('embedding-sentences'),
        ...document.querySelectorAll('input[name="embedding-type"]')
      ];
      controls.forEach(control => {
        control.disabled = true;
      });

      try {
        await renderEmbeddingChart(highlightedIndex);
      } finally {
        embeddingInProgress = false;
        controls.forEach(control => {
          control.disabled = false;
        });
      }
    }

    // WHAT THIS DOES: Reads the active sentence list and removes blank lines and surrounding whitespace
    function readSentenceList() {
      return document.getElementById('embedding-sentences').value
        .split(/\r?\n/)
        .map(sentence => sentence.trim())
        .filter(sentence => sentence.length > 0);
    }

    // WHAT THIS DOES: Creates active-mode vectors, updates pair rankings, and redraws the selected chart
    async function renderEmbeddingChart(highlightedIndex) {
      const chart = document.getElementById('embedding-chart');
      const message = document.getElementById('chart-message');
      const status = document.getElementById('embedding-status');
      const customResults = document.getElementById('custom-sentence-results');
      const pairsMessage = document.getElementById('closest-pairs-message');
      const pairsTable = document.getElementById('closest-pairs-table-container');
      message.style.display = 'none';
      status.style.display = 'none';
      status.classList.remove('error');
      chart.style.display = 'block';
      customResults.style.display = 'none';
      document.getElementById('embedding-summary').style.display = 'none';
      document.getElementById('export-embeddings-btn').disabled = true;
      latestEmbeddingData = null;
      pairsTable.style.display = 'none';
      document.getElementById('closest-pairs-table-body').innerHTML = '';
      pairsMessage.textContent = 'Generate embeddings to see the closest pairs.';
      pairsMessage.style.display = 'block';

      if (typeof Plotly !== 'undefined') {
        Plotly.purge(chart);
      }

      const sentences = readSentenceList();

      if (sentences.length < 4) {
        chart.style.display = 'none';
        message.textContent = 'Enter at least four non-empty sentences to create the chart.';
        message.style.display = 'block';
        document.getElementById('closest-pairs-table-body').innerHTML = '';
        document.getElementById('closest-pairs-table-container').style.display = 'none';
        pairsMessage.textContent = 'Enter at least four non-empty sentences to compare similarity.';
        pairsMessage.style.display = 'block';
        return;
      }

      if (typeof Plotly === 'undefined') {
        chart.style.display = 'none';
        message.textContent = 'The chart library could not be loaded. Check your internet connection and try again.';
        message.style.display = 'block';
        pairsMessage.textContent = 'The chart library could not be loaded, so similarities could not be calculated.';
        return;
      }

      let activeMode = document.querySelector('input[name="embedding-type"]:checked').value;
      let vectors = [];
      let usedFallback = false;
      if (activeMode === 'meaning') {
        status.textContent = 'Loading Transformers.js and the meaning model...';
        status.style.display = 'block';
        try {
          const embedder = await loadMeaningModel(status);
          for (let i = 0; i < sentences.length; i++) {
            status.textContent = `Generating meaning embeddings (${i + 1} of ${sentences.length})...`;
            const output = await embedder(sentences[i], { pooling: 'mean', normalize: true });
            const vector = Array.from(output.data);
            if (vector.length === 0) {
              throw new Error('The meaning model returned an empty embedding.');
            }
            vectors.push(vector);
          }
        } catch (error) {
          status.textContent = 'Meaning mode needs the page served over http - using spelling-based instead';
          status.classList.add('error');
          vectors = sentences.map(createSpellingVector);
          activeMode = 'spelling';
          usedFallback = true;
        }
      } else {
        vectors = sentences.map(createSpellingVector);
      }

      const coordinates = vectors.map(projectToThreeDimensions);
      latestEmbeddingData = {
        sentences: sentences,
        vectors: vectors,
        coordinates: coordinates,
        mode: activeMode,
        highlightedIndex: highlightedIndex
      };

      try {
        await renderActiveChart();
      } catch (error) {
        chart.style.display = 'none';
        status.style.display = 'none';
        pairsMessage.textContent = 'The chart could not be displayed. Check the browser console and try again.';
        message.textContent = 'The chart could not be displayed. Check the browser console and try again.';
        message.style.display = 'block';
        throw error;
      }

      renderClosestPairs(sentences, vectors);
      renderEmbeddingSummary(sentences, vectors, activeMode, usedFallback);
      document.getElementById('export-embeddings-btn').disabled = false;
      if (highlightedIndex !== null) {
        renderCustomSentenceResults(sentences, vectors, highlightedIndex);
      }
      if (activeMode === 'meaning') {
        status.style.display = 'none';
      }
    }

    // WHAT THIS DOES: Validates, tokenizes, appends, embeds, and highlights a user-entered sentence
    async function addCustomSentence() {
      if (embeddingInProgress) {
        return;
      }

      const input = document.getElementById('custom-sentence-input');
      const message = document.getElementById('custom-sentence-message');
      const sentence = input.value.trim();
      message.style.display = 'none';

      if (sentence.length === 0) {
        message.textContent = 'Enter a word or sentence first.';
        message.style.display = 'block';
        return;
      }

      if (!vocabularyBuilt) {
        message.textContent = 'Build the vocabulary first by clicking "Build vocabulary".';
        message.style.display = 'block';
        return;
      }

      tokenizeSentence(sentence);

      const sentenceList = document.getElementById('embedding-sentences');
      const sentences = readSentenceList();
      const highlightedIndex = sentences.length;
      sentences.push(sentence);
      sentenceList.value = sentences.join('\n');

      await embedAndPlotSentences(highlightedIndex);
    }

    // WHAT THIS DOES: Displays the vocabulary size and populates the scrollable table with ID, token, and count
    function renderVocabulary() {
      const vocabDisplay = document.getElementById('vocab-display');
      const vocabSizeText = document.getElementById('vocab-size');
      const tableBody = document.getElementById('vocab-table-body');
      const slider = document.getElementById('vocab-limit-slider');
      const total = sortedVocabularyTokens.length;
      const keptCount = total === 0 ? 0 : Number(slider.value);

      vocabulary = [{ id: 0, token: '[UNK]', count: 0 }];
      for (const [index, item] of sortedVocabularyTokens.slice(0, keptCount).entries()) {
        vocabulary.push({
          id: index + 1,
          token: item.token,
          count: item.count
        });
      }

      document.getElementById('words-kept-count').textContent = keptCount;
      document.getElementById('words-total-count').textContent = total;

      // Show the total vocabulary size including [UNK]
      vocabSizeText.textContent = `Vocabulary size: ${vocabulary.length} (including [UNK])`;

      // Clear any previous table rows
      tableBody.innerHTML = '';

      // Add a table row for each token in the vocabulary
      for (const item of vocabulary) {
        const row = document.createElement('tr');

        const idCell = document.createElement('td');
        idCell.textContent = item.id;
        row.appendChild(idCell);

        const tokenCell = document.createElement('td');
        tokenCell.textContent = item.token;
        row.appendChild(tokenCell);

        const countCell = document.createElement('td');
        countCell.textContent = item.count;
        row.appendChild(countCell);

        tableBody.appendChild(row);
      }

      // Make the vocabulary results section visible
      vocabDisplay.style.display = 'block';
    }

    // WHAT THIS DOES: Attaches the click event listener to the "Build vocabulary" button
    document.addEventListener('DOMContentLoaded', () => {
      const buildVocabButton = document.getElementById('build-vocab-btn');
      buildVocabButton.addEventListener('click', buildVocabulary);
      document.getElementById('vocab-limit-slider').addEventListener('input', () => {
        renderVocabulary();
        if (hasTokenizedSentence) {
          tokenizeSentence(lastTokenizedSentence);
        }
      });
      document.getElementById('tokenize-btn').addEventListener('click', () => tokenizeSentence());
      document.getElementById('embed-plot-btn').addEventListener('click', () => embedAndPlotSentences());
      document.getElementById('add-custom-sentence-btn').addEventListener('click', addCustomSentence);
      document.getElementById('chart-label-toggle').addEventListener('change', updatePointLabels);
      document.getElementById('export-embeddings-btn').addEventListener('click', exportEmbeddingCsv);
      document.getElementById('chart-view-3d-btn').addEventListener('click', () => setChartView('3d'));
      document.getElementById('chart-view-similarity-btn').addEventListener('click', () => setChartView('similarity'));
    });
