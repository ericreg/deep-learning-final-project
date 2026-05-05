import json
import re
from collections.abc import Sequence
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pymupdf
import seaborn as sns
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from transformers import pipeline


SURPRISAL_MODELS = [
    "all-MiniLM-L6-v2",  # 90 MB, 256 / 512, default
    "all-mpnet-base-v2",  # 420 MB, 512, high-accuracy general English text
    "BAAI/bge-base-en-v1.5",  # 440 MB, maximum leaderboard performance
    "nomic-ai/nomic-embed-text-v1.5",  # 550 MB, 8,192 tokens, long chunks
]
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
DEFAULT_RESULTS_DIR = Path("results")
PUNCTUATION_TRANSLATION = str.maketrans(
    {
        "\u00a0": " ",
        "\u1680": " ",
        "\u2000": " ",
        "\u2001": " ",
        "\u2002": " ",
        "\u2003": " ",
        "\u2004": " ",
        "\u2005": " ",
        "\u2006": " ",
        "\u2007": " ",
        "\u2008": " ",
        "\u2009": " ",
        "\u200a": " ",
        "\u202f": " ",
        "\u205f": " ",
        "\u3000": " ",
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u2032": "'",
        "\u2035": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2033": '"',
        "\u2036": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u2026": "...",
        "\u2022": "*",
        "\u00b7": "*",
        "\u2044": "/",
        "\u2215": "/",
    }
)


def extract_text_from_pdf(pdf_path):
    """
    Opens a PDF file and extracts text from every page.
    """
    page_content = []

    try:
        with pymupdf.open(pdf_path) as doc:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)

                page_text = page.get_text().strip()
                if page_text:
                    page_content.append(page.get_text())

        return page_content

    except Exception:
        raise



def clean_text(raw: str) -> str:
    text = raw.translate(PUNCTUATION_TRANSLATION)

    # Collapse spaced ellipses like ". . ." or ".  .  ." into "...".
    text = re.sub(r"\.\s*\.\s*\.", "...", text)

    # Remove the extra space before an ellipsis.
    text = re.sub(r"\s+\.\.\.", "...", text)

    # Remove spaces before common punctuation marks.
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    # Collapse three or more newlines into a paragraph break.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove bare page numbers that appear alone on a line.
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)

    # Collapse runs of spaces and tabs into a single space.
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def chunk_text(corpus: list[list[str]], chunk_size: int = 300) -> list[str]:
    """
    Breaks down a list of strings into smaller chunks of a specified size.
    """
    chunks = []
    words = []

    for page in corpus:
        words += page

        if len(words) >= chunk_size:
            while len(words) >= chunk_size:
                chunks.append(" ".join(words[:chunk_size]))
                words = words[chunk_size:]

    return chunks


def load_pdf_chunks(pdf_path: str = "hp1.pdf", chunk_size: int = 150) -> list[str]:
    """Extract text from a PDF and return fixed-size word chunks."""
    pages = extract_text_from_pdf(pdf_path)
    corpus = [clean_text(page).split() for page in pages]
    return chunk_text(corpus, chunk_size=chunk_size)


def model_safe_name(model_name: str) -> str:
    """Create a filesystem-safe name for model-specific output files."""
    return model_name.replace("/", "_")


def output_path(results_dir: str | Path, filename: str) -> Path:
    """Create the results directory if needed and return an output file path."""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    return results_path / filename


def resolve_device() -> str:
    """Detect the best available device for sentence-transformer embeddings."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"

    print("No GPU detected. Running on CPU, which may be very slow!")
    return "cpu"


class SurprisalAnalysis:
    """Embedding-based semantic surprisal analysis for text chunks."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 256,
        show_progress_bar: bool = True,
        device: str | None = None,
        results_dir: str | Path = DEFAULT_RESULTS_DIR,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.device = device or resolve_device()
        self.results_dir = results_dir
        self._model = None

    @property
    def model(self):
        if self._model is None:
            print(f"Loading model on: {self.device}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def score_consecutive(self, chunks: Sequence[str]) -> list[float]:
        """
        Return each chunk's cosine distance from the preceding chunk.
        """
        if not chunks:
            return []

        embeddings = self.model.encode(
            chunks,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
        )

        surprisal_scores = [0.0]
        for i in range(1, len(embeddings)):
            vector_current = embeddings[i]
            vector_previous = embeddings[i - 1]
            surprisal_scores.append(cosine(vector_previous, vector_current))

        return surprisal_scores

    def score_pairwise(self, chunks: Sequence[str]) -> list[float]:
        """
        Return each chunk's mean cosine distance from all other chunks.
        """
        if not chunks:
            return []

        embeddings = self.model.encode(
            chunks,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=True,
            device=self.device,
        )

        embeddings = F.normalize(embeddings, p=2, dim=1)
        sim = embeddings @ embeddings.T
        dist = 1.0 - sim

        n = dist.shape[0]
        avg = dist.sum(dim=1) / max(n - 1, 1)

        return avg.cpu().tolist()

    def write_results(self, chunks: Sequence[str], scores: Sequence[float], top_n: int = 5) -> str:
        """Write ranked surprisal chunks to a model-specific JSON file."""
        filtered_indexed = self._filter_outliers(scores)
        ranked = sorted(filtered_indexed, key=lambda x: x[1], reverse=True)
        output_file = output_path(
            self.results_dir,
            f"surprisal_results_{model_safe_name(self.model_name)}.json",
        )

        def result_entry(rank: int, chunk_index: int, score: float) -> dict:
            return {
                "rank": rank,
                "chunk_index": chunk_index,
                "surprisal": float(score),
                "chunk": chunks[chunk_index],
            }

        results = {
            "model_name": self.model_name,
            "metric": "surprisal",
            "top_n": top_n,
            "score_count": len(scores),
            "filtered_score_count": len(filtered_indexed),
            "top": [result_entry(rank, i, score) for rank, (i, score) in enumerate(ranked[:top_n], 1)],
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            f.write("\n")

        return str(output_file)

    def plot_distribution(self, scores: Sequence[float]) -> str:
        """Save a histogram of filtered surprisal scores."""
        filtered_indexed = self._filter_outliers(scores)
        output_file = output_path(
            self.results_dir,
            f"surprisal_distribution_{model_safe_name(self.model_name)}.png",
        )

        fig, ax = plt.subplots()
        sns.histplot([x[1] for x in filtered_indexed], bins=30, kde=True, ax=ax)
        ax.set_xlabel("Surprisal Score (cosine distance)")
        ax.set_ylabel("Count")
        ax.set_title(f"Surprisal Score Distribution - {self.model_name}")
        fig.tight_layout()
        fig.savefig(output_file)
        plt.show()
        plt.close(fig)

        return str(output_file)

    def run(self, chunks: Sequence[str], top_n: int = 5, pairwise: bool = True) -> list[float]:
        """Score chunks, write ranked examples, and save a distribution plot."""
        scores = self.score_pairwise(chunks) if pairwise else self.score_consecutive(chunks)
        if len(scores) == 0:
            return scores

        self.write_results(chunks, scores, top_n=top_n)
        self.plot_distribution(scores)
        return scores

    @staticmethod
    def _filter_outliers(scores: Sequence[float]) -> list[tuple[int, float]]:
        if len(scores) == 0:
            return []

        p_bottom, p_top = np.percentile(scores, [0.5, 99.5])
        return [(i, s) for i, s in enumerate(scores) if p_bottom <= s <= p_top]


class EmotionalAnalysis:
    """Emotion-vector volatility analysis for text chunks."""

    def __init__(
        self,
        model_name: str = EMOTION_MODEL,
        batch_size: int = 64,
        max_length: int = 512,
        device_id: int | None = None,
        results_dir: str | Path = DEFAULT_RESULTS_DIR,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device_id = device_id if device_id is not None else (0 if torch.cuda.is_available() else -1)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.results_dir = results_dir
        self._classifier = None

    @property
    def classifier(self):
        if self._classifier is None:
            self._classifier = pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None,
                device=self.device_id,
            )
        return self._classifier

    def score_consecutive(self, chunks: Sequence[str]) -> tuple[list[float], list[str]]:
        """
        Score each chunk and return emotion volatility from the preceding chunk.
        """
        if not chunks:
            return [], []

        emotion_vectors, dominant_emotions = self._emotion_vectors(chunks)

        volatility_scores = [0.0]
        for i in range(1, len(emotion_vectors)):
            vec_current = emotion_vectors[i]
            vec_previous = emotion_vectors[i - 1]
            volatility_scores.append(cosine(vec_previous, vec_current))

        return volatility_scores, dominant_emotions

    def score_pairwise(self, chunks: Sequence[str]) -> tuple[list[float], list[str]]:
        """
        Return each chunk's mean emotion-vector distance from all other chunks.
        """
        if not chunks:
            return [], []

        vectors, dominant_emotions = self._emotion_vectors(chunks)
        emotion_vectors = torch.tensor(vectors, dtype=torch.float32, device=self.device)

        emotion_vectors = F.normalize(emotion_vectors, p=2, dim=1)
        sim = emotion_vectors @ emotion_vectors.T
        dist = (1.0 - sim).clamp(min=0.0)

        n = dist.shape[0]
        volatility = dist.sum(dim=1) / max(n - 1, 1)

        return volatility.cpu().tolist(), dominant_emotions

    def write_results(
        self,
        chunks: Sequence[str],
        scores: Sequence[float],
        dominant_emotions: Sequence[str],
        top_n: int = 5,
    ) -> str:
        """Write ranked emotion-volatility chunks to a model-specific JSON file."""
        filtered_indexed = self._filter_outliers(scores)
        ranked = sorted(filtered_indexed, key=lambda x: x[1], reverse=True)
        output_file = output_path(
            self.results_dir,
            f"volatility_results_{model_safe_name(self.model_name)}.json",
        )

        def result_entry(rank: int, chunk_index: int, score: float) -> dict:
            return {
                "rank": rank,
                "chunk_index": chunk_index,
                "volatility": float(score),
                "dominant_emotion": dominant_emotions[chunk_index],
                "chunk": chunks[chunk_index],
            }

        results = {
            "model_name": self.model_name,
            "metric": "volatility",
            "top_n": top_n,
            "score_count": len(scores),
            "filtered_score_count": len(filtered_indexed),
            "top": [result_entry(rank, i, score) for rank, (i, score) in enumerate(ranked[:top_n], 1)],
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            f.write("\n")

        return str(output_file)

    def plot_distribution(self, scores: Sequence[float]) -> str:
        """Save a histogram of filtered emotion-volatility scores."""
        filtered_indexed = self._filter_outliers(scores)
        output_file = output_path(
            self.results_dir,
            f"volatility_distribution_{model_safe_name(self.model_name)}.png",
        )

        fig, ax = plt.subplots()
        sns.histplot([x[1] for x in filtered_indexed], bins=30, kde=True, ax=ax)
        ax.set_xlabel("Volatility Score (cosine distance)")
        ax.set_ylabel("Count")
        ax.set_title(f"Volatility Score Distribution - {self.model_name}")
        fig.tight_layout()
        fig.savefig(output_file)
        plt.show()
        plt.close(fig)

        return str(output_file)

    def run(self, chunks: Sequence[str], top_n: int = 5, pairwise: bool = True) -> tuple[list[float], list[str]]:
        """Score chunks, write ranked examples, and save a distribution plot."""
        scores, dominant_emotions = self.score_pairwise(chunks) if pairwise else self.score_consecutive(chunks)
        if len(scores) == 0:
            return scores, dominant_emotions

        self.write_results(chunks, scores, dominant_emotions, top_n=top_n)
        self.plot_distribution(scores)
        return scores, dominant_emotions

    def _emotion_vectors(self, chunks: Sequence[str]) -> tuple[list[list[float]], list[str]]:
        raw_scores = self.classifier(
            chunks,
            batch_size=self.batch_size,
            truncation=True,
            max_length=self.max_length,
        )

        dominant_emotions = []
        vectors = []
        for chunk_scores in raw_scores:
            chunk_scores.sort(key=lambda x: x["label"])
            vectors.append([d["score"] for d in chunk_scores])
            dominant_emotions.append(max(chunk_scores, key=lambda x: x["score"])["label"])

        return vectors, dominant_emotions

    @staticmethod
    def _filter_outliers(scores: Sequence[float]) -> list[tuple[int, float]]:
        if len(scores) == 0:
            return []

        p_bottom, p_top = np.percentile(scores, [0.5, 99.5])
        return [(i, s) for i, s in enumerate(scores) if p_bottom <= s <= p_top]


def find_semantic_surprisal(chunks, model_name):
    """
    Takes a list of text chunks in chronological order and returns each
    chunk's semantic surprisal score against the previous chunk.
    """
    return SurprisalAnalysis(model_name).score_consecutive(chunks)


def find_semantic_surprisal_pairwise(chunks, model_name):
    """
    Takes a list of text chunks and returns each chunk's mean semantic
    surprisal score against all other chunks.
    """
    return SurprisalAnalysis(model_name).score_pairwise(chunks)


def find_emotion_volatility(chunks, model_name):
    """
    Scores each chunk for emotions and returns volatility against the previous chunk.
    """
    return EmotionalAnalysis(model_name).score_consecutive(chunks)


def find_emotion_volatility_pairwise(chunks, model_name):
    """
    Scores each chunk for emotions and returns mean volatility against all other chunks.
    """
    return EmotionalAnalysis(model_name).score_pairwise(chunks)


def emotional_analysis(
    pdf_path: str = "hp1.pdf",
    chunk_size: int = 150,
    top_n: int = 5,
    pairwise: bool = True,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
):
    chunks = load_pdf_chunks(pdf_path, chunk_size=chunk_size)
    print("Evaluating model:", EMOTION_MODEL)
    EmotionalAnalysis(EMOTION_MODEL, results_dir=results_dir).run(
        chunks,
        top_n=top_n,
        pairwise=pairwise,
    )


def surprisal_analysis(
    pdf_path: str = "hp1.pdf",
    chunk_size: int = 150,
    top_n: int = 5,
    pairwise: bool = True,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
):
    chunks = load_pdf_chunks(pdf_path, chunk_size=chunk_size)

    for model_name in SURPRISAL_MODELS:
        print("Evaluating model:", model_name)
        SurprisalAnalysis(model_name, results_dir=results_dir).run(
            chunks,
            top_n=top_n,
            pairwise=pairwise,
        )


@click.command()
@click.option(
    "--pdf-path",
    default="docs/hp.pdf",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False, path_type=str),
    help="PDF file to analyze.",
)
@click.option(
    "--analysis",
    type=click.Choice(["all", "emotion", "surprisal"]),
    default="all",
    show_default=True,
    help="Analysis pipeline to run.",
)
@click.option(
    "--chunk-size",
    default=150,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of words per chunk.",
)
@click.option(
    "--top-n",
    default=5,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of top-ranked chunks to write to JSON.",
)
@click.option(
    "--pairwise/--consecutive",
    default=True,
    show_default=True,
    help="Compare each chunk to all chunks, or only to the previous chunk.",
)
@click.option(
    "--results-dir",
    default=str(DEFAULT_RESULTS_DIR),
    show_default=True,
    type=click.Path(file_okay=False, path_type=str),
    help="Directory for JSON results and distribution plots.",
)
def cli(
    pdf_path: str,
    analysis: str,
    chunk_size: int,
    top_n: int,
    pairwise: bool,
    results_dir: str,
):
    """Run text surprisal and emotion analysis over a PDF."""
    chunks = load_pdf_chunks(pdf_path, chunk_size=chunk_size)
    click.echo(f"Loaded {len(chunks)} chunks from {pdf_path}")
    click.echo(f"Writing results to {results_dir}")

    if analysis in {"all", "emotion"}:
        click.echo(f"Evaluating model: {EMOTION_MODEL}")
        EmotionalAnalysis(EMOTION_MODEL, results_dir=results_dir).run(
            chunks,
            top_n=top_n,
            pairwise=pairwise,
        )

    if analysis in {"all", "surprisal"}:
        for model_name in SURPRISAL_MODELS:
            click.echo(f"Evaluating model: {model_name}")
            SurprisalAnalysis(model_name, results_dir=results_dir).run(
                chunks,
                top_n=top_n,
                pairwise=pairwise,
            )


if __name__ == "__main__":
    cli()
