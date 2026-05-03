import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np

import re
import nltk
from scipy.spatial.distance import cosine

# Run these once to download the necessary data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

import seaborn as sns
import matplotlib.pyplot as plt
from transformers import pipeline
import torch

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


def extract_text_from_pdf(pdf_path):
    """
    Opens a PDF file and extracts text from every page.
    """
    page_content = []
    
    try:
        # Open the PDF document
        with fitz.open(pdf_path) as doc:
            for page_num in range(len(doc)):
                # Load the individual page
                page = doc.load_page(page_num)

                page_text = page.get_text().strip()
                if page_text:  # Only add non-empty pages
                    page_content.append(page.get_text())
        
        return page_content

    except Exception as e:
        return f"An error occurred: {e}"


def clean_text(text):

    # Remove non-alphabetical characters (punctuation/numbers)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Lowercase and split into words
    return text.lower().split()


def chunk_text(corpus: list[list[str]], chunk_size: int = 300) -> list [str]:
    """
    Breaks down a list of strings into smaller chunks of a specified size.
    """

    chunks = []
    words = []

    for page in corpus:

        # get the words in the current page
        words += page

        if len(words) >= chunk_size:

            # if the current text has enough words. keep add chunks until we have less than the chunk size
            while len(words) >= chunk_size:
                # add the first chunk_size words as a chunk
                chunks.append(" ".join(words[:chunk_size]))

                # remove the chunked words from the current text
                words = words[chunk_size:]

    return chunks


def find_semantic_surprisal(chunks, model_name):
    """
    Takes a list of text chunks (strings) in chronological order 
    and returns the 'Surprisal Score' for each chunk.
    """

    # all-MiniLM-L6-v2,90 MB,256 / 512,"Rapid prototyping, laptops, CPU-only"
    # all-mpnet-base-v2,420 MB,512,High-accuracy general English text
    # BAAI/bge-base-en-v1.5,440 MB,512,Maximum leaderboard performance
    # nomic-embed-text-v1.5,550 MB,"8,192",Long chunks (multi-paragraph or whole chapters)

    # 1. Automatically detect the best available hardware
    if torch.cuda.is_available():
        device = 'cuda'  # NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = 'mps'   # Apple Silicon (M1/M2/M3/M4)
    else:
        device = 'cpu'   # Fallback

    print(f"Loading model on: {device}")

    # Load a fast, lightweight BERT model
    model = SentenceTransformer(model_name, device=device)

    # 3. Encode in batches to maximize GPU parallel processing
    # Adjust batch_size (32, 64, 128) based on your GPU's VRAM
    embeddings = model.encode(
        chunks, 
        batch_size=256, 
        show_progress_bar=True # Helpful to see the speed
    )
        
    surprisal_scores = [0.0] # The first chunk has nothing to compare to
    
    # 2. Compare each chunk to the one right before it
    for i in range(1, len(embeddings)):
        vector_current = embeddings[i]
        vector_previous = embeddings[i-1]
        
        # Cosine distance ranges from 0 (identical) to 2 (opposites)
        # Higher distance = Higher surprisal = Potential Plot Twist
        distance = cosine(vector_previous, vector_current)
        surprisal_scores.append(distance)
        
    return surprisal_scores



def find_semantic_surprisal_pairwise(chunks, model_name):
    """
    Takes a list of text chunks (strings) in chronological order
    and returns the 'Surprisal Score' for each chunk: the mean
    cosine distance to all preceding chunks.
    """
    # 1. Detect best available hardware
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Loading model on: {device}")

    model = SentenceTransformer(model_name, device=device)

    # 2. Encode and keep tensors on device (don't convert to numpy)
    embeddings = model.encode(
        chunks,
        batch_size=256,
        show_progress_bar=True,
        convert_to_tensor=True,
        device=device,
    )

    # 3. Normalize once, then a single matmul gives the full NxN
    #    cosine-similarity matrix on the GPU.
    embeddings = F.normalize(embeddings, p=2, dim=1)
    sim = embeddings @ embeddings.T          # (N, N) cosine similarities
    dist = 1.0 - sim                         # cosine distances
    # n = dist.shape[0]

    # # 4. For row i, average dist[i, 0:i]. Build a strictly-lower-triangular
    # #    mask and use it to compute row-wise means over the prior chunks.
    # idx = torch.arange(n, device=device)
    # mask = idx.unsqueeze(1) > idx.unsqueeze(0)        # (N, N), True where j < i
    # counts = mask.sum(dim=1).clamp(min=1)             # row 0 has 0 priors
    # sums = (dist * mask).sum(dim=1)
    # avg = sums / counts
    # avg[0] = 0.0                                       # first chunk: nothing to compare
    # return avg.cpu().tolist()

    n = dist.shape[0]
    # Average distance from each chunk to all OTHER chunks (exclude self).
    # dist[i, i] = 0 already, so summing the full row and dividing by (n-1)
    # gives the mean over the n-1 non-self entries.
    sums = dist.sum(dim=1)
    avg = sums / max(n - 1, 1)

    return avg.cpu().tolist()

def find_emotion_volatility(chunks, model_name):
    """
    Scores each chunk for 7 emotions and calculates the volatility 
    (the shift in emotion) from the previous chunk.
    """
    
    # Use GPU if available, otherwise fallback to CPU
    device_id = 0 if torch.cuda.is_available() else -1
    
    # Load the Hartmann Emotion model
    # top_k=None forces the model to return probabilities for ALL 7 emotions, not just the top 1
    classifier = pipeline(
        "text-classification", 
        model=model_name,
        top_k=None, 
        device=device_id
    )

    # Process in batches. Truncation ensures chunks over 512 tokens don't crash the model
    raw_scores = classifier(chunks, batch_size=64, truncation=True, max_length=512, show_progress_bar=True)
    
    emotion_vectors = []
    dominant_emotions = []
    
    for chunk_scores in raw_scores:
        
        # The model returns a list of dicts. We sort them alphabetically by label 
        # (anger, disgust, fear...) so our vectors always align perfectly.
        chunk_scores.sort(key=lambda x: x['label'])
        
        # Extract the probability values (0.0 to 1.0) into a list
        probs = [d['score'] for d in chunk_scores]
        emotion_vectors.append(probs)
        
        # Keep track of the strongest emotion for our output text
        dominant = max(chunk_scores, key=lambda x: x['score'])['label']
        dominant_emotions.append(dominant)
        
    # Calculate Volatility (Cosine distance between N and N-1)
    volatility_scores = [0.0] 
    
    for i in range(1, len(emotion_vectors)):
        vec_current = emotion_vectors[i]
        vec_previous = emotion_vectors[i-1]
        
        distance = cosine(vec_previous, vec_current)
        volatility_scores.append(distance)
        
    return volatility_scores, dominant_emotions



def find_emotion_volatility_pairwise(chunks, model_name):
    """
    Scores each chunk across 7 emotions and returns, for each chunk,
    the mean cosine distance of its emotion vector to every other
    chunk's emotion vector. Also returns the dominant emotion per chunk.
    """
    # 1. Hardware selection
    device_id = 0 if torch.cuda.is_available() else -1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 2. Load Hartmann emotion classifier; top_k=None returns all 7 labels
    classifier = pipeline(
        "text-classification",
        model=model_name,
        top_k=None,
        device=device_id,
    )

    raw_scores = classifier(
        chunks,
        batch_size=64,
        truncation=True,
        max_length=512,
    )

    # 3. Build a (N, 7) tensor of emotion probabilities, with labels in a
    #    consistent order across rows. Track dominant emotion per chunk.
    dominant_emotions = []
    vectors = []
    for chunk_scores in raw_scores:
        chunk_scores.sort(key=lambda x: x['label'])  # alphabetical: anger, disgust, fear, ...
        vectors.append([d['score'] for d in chunk_scores])
        dominant_emotions.append(max(chunk_scores, key=lambda x: x['score'])['label'])

    emotion_vectors = torch.tensor(vectors, dtype=torch.float32, device=device)

    # 4. Pairwise cosine distance via single matmul on normalized rows.
    emotion_vectors = F.normalize(emotion_vectors, p=2, dim=1)
    sim = emotion_vectors @ emotion_vectors.T          # (N, N) cosine sims
    dist = (1.0 - sim).clamp(min=0.0)                  # guard tiny negatives

    # 5. Mean distance from each chunk to all other chunks. Diagonal is 0,
    #    so summing the full row and dividing by (n-1) gives the off-diagonal mean.
    n = dist.shape[0]
    volatility = dist.sum(dim=1) / max(n - 1, 1)

    return volatility.cpu().tolist(), dominant_emotions

def emotional_analysis():
    # Example Usage:
    pages = extract_text_from_pdf("hp1.pdf")

    # clean and normalize the text
    corpus = [x.split() for x in pages]

    # break the text into smaller chunks (e.g., 150 words each)
    chunks = chunk_text(corpus, chunk_size=150)

    # for model_name in models:
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    print("Evaluating model:", model_name)

    # calculate the surprisal scores for each chunk
    # surpisal_scores = find_semantic_surprisal(chunks, model_name)
    volatility_scores, dominant_emotions = find_emotion_volatility_pairwise(chunks, model_name)

    # Filter out outliers using the 0.5 and 99.5 percentiles
    p_bottom, p_top = np.percentile(volatility_scores, [0.5, 99.5])
    filtered_indexed = [(i, s) for i, s in enumerate(volatility_scores) if p_bottom <= s <= p_top]

    # Rank the chunks by volatility score (highest first)
    ranked = sorted(filtered_indexed, key=lambda x: x[1], reverse=True)

    output_file = f"volatility_results_{model_name.replace('/', '_')}.txt"
    with open(output_file, 'w') as f:
        f.write("TOP 5:\n")
        for rank, (i, score) in enumerate(ranked[:5], 1):
            f.write(f"\n[#{rank} | chunk {i} | volatility={score:.4f}]\n")
            f.write(f"Dominant Emotions: {dominant_emotions[i]}\n")
            f.write(chunks[i] + "\n")

        f.write(f"BOTTOM 5:\n")
        for rank, (i, score) in enumerate(ranked[-5:][::-1], 1):
            f.write(f"\n[#{rank} | chunk {i} | volatility={score:.4f}]\n")
            f.write(f"Dominant Emotions: {dominant_emotions[i]}\n") 
            f.write(chunks[i] + "\n")

        # print out the 5 chunks closests to the mean volatility score
        mean_score = sum(volatility_scores) / len(volatility_scores)
        ranked_by_distance_to_mean = sorted(enumerate(volatility_scores), key=lambda x: abs(x[1] - mean_score))

        f.write(f"MEAN 5:\n")
        for rank, (i, score) in enumerate(ranked_by_distance_to_mean[:5], 1):
            f.write(f"\n[#{rank} | chunk {i} | volatility={score:.4f}]\n")
            f.write(f"Dominant Emotions: {dominant_emotions[i]}\n")
            f.write(chunks[i] + "\n")


    # plot the distribution of volatility scores
    fig, ax = plt.subplots()
    sns.histplot([x[1] for x in filtered_indexed], bins=30, kde=True, ax=ax)
    ax.set_xlabel("Volatility Score (cosine distance)")
    ax.set_ylabel("Count")
    ax.set_title(f"Volatility Score Distribution — {model_name}")
    fig.tight_layout()
    fig.savefig("volatility_distribution_{}.png".format(model_name.replace('/', '_')))
    plt.show()
    plt.close(fig)



def surprisal_analysis():
    # Example Usage:
    pages = extract_text_from_pdf("hp1.pdf")

    # clean and normalize the text
    corpus = [x.split() for x in pages]

    # break the text into smaller chunks (e.g., 150 words each)
    chunks = chunk_text(corpus, chunk_size=150)

    models = [
        'all-MiniLM-L6-v2', # 90 MB, 256 / 512, default
        'all-mpnet-base-v2', # 420 MB, 512, High-accuracy general English text
        'BAAI/bge-base-en-v1.5', # 440 MB, 512, Maximum leaderboard performance
        'nomic-ai/nomic-embed-text-v1.5', # 550 MB, 8,192, Long chunks (multi-paragraph or whole chapters)
    ]

    for model_name in models:

        print("Evaluating model:", model_name)

        # calculate the surprisal scores for each chunk
        # surpisal_scores = find_semantic_surprisal(chunks, model_name)
        surprisal_scores = find_semantic_surprisal_pairwise(chunks, model_name)

        # Filter out outliers using the 0.5 and 99.5 percentiles
        p_bottom, p_top = np.percentile(surprisal_scores, [0.5, 99.5])
        filtered_indexed = [(i, s) for i, s in enumerate(surprisal_scores) if p_bottom <= s <= p_top]

        # Rank the chunks by volatility score (highest first)
        ranked = sorted(filtered_indexed, key=lambda x: x[1], reverse=True)

        output_file = f"surprisal_results_{model_name.replace('/', '_')}.txt"
        with open(output_file, 'w') as f:
            f.write("TOP 5:\n")
            for rank, (i, score) in enumerate(ranked[:5], 1):
                f.write(f"\n[#{rank} | chunk {i} | surprisal={score:.4f}]\n")
                f.write(chunks[i] + "\n")

            f.write(f"BOTTOM 5:\n")
            for rank, (i, score) in enumerate(ranked[-5:][::-1], 1):
                f.write(f"\n[#{rank} | chunk {i} | surprisal={score:.4f}]\n")
                f.write(chunks[i] + "\n")

            # print out the 5 chunks closests to the mean surprisal score
            mean_score = sum(surprisal_scores) / len(surprisal_scores)
            ranked_by_distance_to_mean = sorted(enumerate(surprisal_scores), key=lambda x: abs(x[1] - mean_score))

            f.write(f"MEAN 5:\n")
            for rank, (i, score) in enumerate(ranked_by_distance_to_mean[:5], 1):
                f.write(f"\n[#{rank} | chunk {i} | surprisal={score:.4f}]\n")
                f.write(chunks[i] + "\n")


        # plot the distribution of surprisal scores
        fig, ax = plt.subplots()
        sns.histplot([x[1] for x in filtered_indexed], bins=30, kde=True, ax=ax)
        ax.set_xlabel("Surprisal Score (cosine distance)")
        ax.set_ylabel("Count")
        ax.set_title(f"Surprisal Score Distribution — {model_name}")
        fig.tight_layout()
        fig.savefig("surprisal_distribution_{}.png".format(model_name.replace('/', '_')))
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    emotional_analysis()
    surprisal_analysis()
