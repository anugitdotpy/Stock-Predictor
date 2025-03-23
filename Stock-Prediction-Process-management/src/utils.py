import requests
from datetime import datetime
from bs4 import BeautifulSoup
import json
import torch


def detect_topics_in_summary(
    summary_text,
    candidate_topics,
    topic_pipeline,
    score_threshold=0.7
):
    """
    Given a summary string, run zero-shot classification with multi_label=True,
    and return the topics that exceed 'score_threshold'.
    """
    if not summary_text.strip():
        return []
    result = topic_pipeline(
        summary_text,
        candidate_labels=candidate_topics,
        multi_label=True
    )
    topics_selected = []
    while not topics_selected and score_threshold >= 0.3:
      for label, score in zip(result["labels"], result["scores"]):
          if score >= score_threshold:
              topics_selected.append(label)
      score_threshold =- 0.1
    return topics_selected


def tokenize_in_sliding_window(text, tokenizer, window_size=512, overlap=100):
    """
    Splits text into segments of up to 'window_size' tokens,
    with 'overlap' tokens carried over between segments.
    Returns a list of chunk strings.
    """
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_tensors="pt",
        truncation=False
    )
    input_ids = encoded["input_ids"][0].tolist()
    chunks = []
    start = 0
    while start < len(input_ids):
        end = start + window_size
        chunk_ids = input_ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
        start += (window_size - overlap)
    return chunks


def get_finbert_logits(request, text, tokenizer):
    """
    Given text (already <=512 tokens), get FinBERT raw logits (shape [3]).
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = request.app.state.FINBERT(**inputs)
    return outputs.logits[0]


def chunk_text_for_bart(text, tokenizer, max_input_length=1024):
    """_summary_

    Args:
        text (_type_): _description_
        tokenizer (_type_): _description_
        max_input_length (int, optional): _description_. Defaults to 1024.

    Returns:
        _type_: _description_
    """
    encoding = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    tokens = encoding["input_ids"][0].tolist()
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_input_length
        chunk_ids = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
        start += max_input_length
    return chunks


def summarize_text_bart(text, summarizer):
    """
    Summarize the given text using Bart-Large-CNN pipeline.
    Returns a concise string summary.
    """
    if not text.strip():
        return ""
    bart_tokenizer = summarizer.tokenizer
    chunked_texts = chunk_text_for_bart(text, bart_tokenizer, 1024)
    all_summaries = []
    for chunk in chunked_texts:
        sum_dict = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
        all_summaries.append(sum_dict[0]["summary_text"])
    return " ".join(all_summaries) if all_summaries else ""


# def get_raw_logits(request, text, tokenizer):
#     """
#     Encode the text (up to 512 tokens) and get the raw logits from FinBERT.
#     """
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
#     inputs = {k: v.to("cuda") for k, v in inputs.items()}
#     with torch.no_grad():
#         outputs = request.app.state.FINBERT(**inputs)
#     return outputs.logits[0]


def analyze_article_with_summarization(request, title, text, w_summary=2.0, w_article=1.0):
    """
    1) Summarize the text with Bart.
    2) Get raw logits of summary with FinBERT.
    3) Chunk entire article -> sum chunk logits with FinBERT.
    4) Weighted combination of summary logits + article logits.
    5) Softmax => final label.

    Returns (final_label, final_probs, combined_logits, summary_text)
    """
    text = text.strip()
    title = title.strip()
    if not title:
      title = "article data."
    if not title.endswith("."):
      title += "."
    if not text:
        return ("neutral", [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], "", [])
    text_to_be_summarized = title + " " + text
    summary_text = summarize_text_bart(text_to_be_summarized, request.app.state.SUMMARIZER)

    summary_logits = torch.zeros(3).to("cuda")
    if summary_text.strip():
        summary_logits = get_finbert_logits(request, summary_text, request.app.state.FINBERT_TOKENIZER)
    topics = detect_topics_in_summary(summary_text, request.app.state.TOPICS, request.app.state.ZSCLASSIFIER, score_threshold=0.7)
    chunks = tokenize_in_sliding_window(text, request.app.state.FINBERT_TOKENIZER, window_size=512, overlap=100)
    article_logits_sum = torch.zeros(3).to("cuda")
    for chunk_text in chunks:
        article_logits_sum += get_finbert_logits(request, chunk_text, request.app.state.FINBERT_TOKENIZER)

    combined_logits = (w_summary * summary_logits) + (w_article * article_logits_sum)

    final_probs = torch.softmax(combined_logits, dim=0)
    pred_label_id = torch.argmax(final_probs).item()
    final_label = request.app.state.LABEL_MAP[pred_label_id]

    return (
        final_label,
        final_probs.tolist(),
        combined_logits.tolist(),
        summary_text,
        topics
    )


def generate_comparative_analysis_llm(request, analysis_results):
    """
    Uses an LLM (via 'chain') to produce a single textual "comparative analysis"
    comparing articles in detail, including pairwise comparisons.
    """
    example_comparison = (
        "Example pairwise comparison:\n"
        "Article A vs. Article B:\n"
        " - Coverage Differences: Article A focuses on Tesla's new product launch, "
        "while Article B is more about regulatory challenges.\n"
        " - Sentiment Differences: Article A is optimistic (positive), "
        "Article B is cautious (neutral to negative).\n"
        " - Overlapping Topics: Both mention Tesla's competitive strategy. "
        "Unique topics include autopilot features in A vs. international policy in B.\n"
        " - Conclusion: Both converge on Tesla's growth potential but differ in tone "
        "due to Article B's emphasis on regulation.\n\n"
    )

    question = (
        "You are an expert financial and market analyst. You have multiple articles, "
        "each with a summary, topics, and sentiment (positive, negative, or neutral). "
        "Please produce a thorough comparative report covering all articles.\n\n"
        + example_comparison
        + "Here are the actual articles:\n\n"
    )

    for i, result in enumerate(analysis_results):
        question += f"Article {i+1}:\n"
        question += f"Title: {result['title']}\n"
        question += f"Summary: {result['summary']}\n"
        question += f"Topics: {', '.join(result['topics']) if result['topics'] else 'None'}\n"
        question += f"Sentiment: {result['final_label']}\n\n"

    question += (
        "Now, please compare these articles in detail. For **each pair** of articles "
        "(e.g., Article 1 vs. Article 2, Article 1 vs. Article 3, etc.), discuss:\n"
        " - Key coverage differences (timeframe, scope, or emphasis)\n"
        " - Sentiment similarities or differences\n"
        " - Overlapping vs. unique topics\n"
        " - Notable nuances (e.g., different angles on the same event)\n\n"
        "Then, provide an **overall conclusion** that brings everything together, "
        "focusing on major themes, any conflicting viewpoints, and possible implications "
        "for investors or the broader market.\n\n"
        "Use a professional, succinct tone, and avoid generic or repetitive statements.\n"
        "Begin your comparative analysis now:\n"
    )
    response = request.app.state.CONVERSER.invoke({"question": question})
    return response


def analyze_finbert_sentiment(request, articles):
    """
    For each article, we:
      - Summarize text with Bart
      - Get summary logits
      - Chunk article text -> sum chunk logits
      - Weight summary vs article
      - Produce final sentiment label (softmax once)

    Returns a list of dicts
    """
    results = []
    for article in articles:
        text = article.get("text", "")
        title = article.get("title", "")
        final_label, final_probs, combined_logits, summary, topics = analyze_article_with_summarization(
            request,
            title,
            text,
            w_summary=2.0,
            w_article=1.0
        )

        results.append({
            "title": article.get("title"),
            "url": article.get("url"),
            "final_label": final_label,
            "final_probs": final_probs,
            "combined_logits": combined_logits,
            "summary": summary,
            "topics": topics
        })
    sentiments = [r["final_label"] for r in results]
    pos_count = sum(s == "positive" for s in sentiments)
    neg_count = sum(s == "negative" for s in sentiments)
    neu_count = sum(s == "neutral"  for s in sentiments)

    sentiment_distribution = {
        "Positive": pos_count,
        "Negative": neg_count,
        "Neutral": neu_count
    }
    all_topics_list = [set(r["topics"]) for r in results]
    if len(all_topics_list) > 1:
        common_topics = set.intersection(*all_topics_list)
        unique_topics_list = []
        for i, topics_set in enumerate(all_topics_list):
            others = set.union(*(all_topics_list[:i]+all_topics_list[i+1:]))
            unique_for_this = topics_set - others
            unique_topics_list.append(list(unique_for_this))
    else:
        common_topics = all_topics_list[0] if all_topics_list else set()
        unique_topics_list = [list(all_topics_list[0])] if all_topics_list else [[]]
    final_response = {
        "articles": results,
        "comparative sentiment score":{
          "sentiment distribution": sentiment_distribution,
          "Topic Overlap": {
            "Common Topics": common_topics,
            "Unique Topics": unique_topics_list,
          }
        }
    }
    return final_response


def parse_and_aggregate_articles(request, articles):
    """
    Given a list of article dicts (each having at least "title" and "url"),
    this function fetches each article's webpage, attempts to parse the main
    text content, and prints it to the console.
    """
    processed_articles = []
    for idx, article in enumerate(articles, start=1):
        single_article = {}
        url = article.get("url")
        title = article.get("title", "No Title")
        single_article["title"] = title
        single_article["url"] = url

        if not url:
            print("[!] No URL found for this article.\n")
            continue

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"[!] Error fetching article from {url}: {e}\n")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        full_text = None

        json_ld_tags = soup.find_all("script", attrs={"type": "application/ld+json"})
        for script_tag in json_ld_tags:
            try:
                data = json.loads(script_tag.string)
            except (json.JSONDecodeError, TypeError):
                continue

            if isinstance(data, dict):
                if 'articleBody' in data:
                    full_text = data['articleBody']
                    break

            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'articleBody' in item:
                        full_text = item['articleBody']
                        break
                if full_text:
                    break

        if not full_text:

            main_content_div = soup.find("div", {"class": "storypage-content"})
            if main_content_div:
                full_text = main_content_div.get_text(separator="\n", strip=True)
            else:
                pass

        if full_text:
          single_article["text"] = full_text
        else:
            print("[!] Could not extract the full article text.")
        processed_articles.append(single_article)
    return analyze_finbert_sentiment(request, processed_articles)


def scrape_indian_express_api_fil(
    request,
    search_query=None,
    author=None,
    from_date=None,    # Expected format: "YYYY-MM-DD"
    to_date=None,      # Expected format: "YYYY-MM-DD"
    max_articles=10,
):
    """
    Scrapes article information from the New Indian Express advanced-search API.
    It can optionally filter results by a search query (keyword), an author's name,
    and/or a date range (from_date, to_date).  If no filter is specified, it
    returns the same full list (up to max_articles) as a normal advanced search.

    :param search_query: (str) If given, used as `q` parameter to filter by headline/content
    :param author: (str) If given, only keep articles that match the author name
    :param from_date: (str) If given, keep articles published >= this date (YYYY-MM-DD)
    :param to_date: (str) If given, keep articles published <= this date (YYYY-MM-DD)
    :param max_articles: (int) Max number of articles to return
    :return: List of dictionaries, each with info like 'title', 'url', 'snippet', 'date', 'author'
    """
    base_url = "https://www.newindianexpress.com/api/v1/advanced-search"

    fields = (
        "alternative,slug,metadata,story-template,story-content-id,id,headline,"
        "hero-image-s3-key,hero-image-metadata,sections,tags,author-name,author-id,"
        "authors,created-at,first-published-at,published-at,last-published-at,url,"
        "subheadline,read-time,access,hero-image-caption,hero-image-alt-text"
    )

    limit = 10
    offset = 0
    collected = []

    from_dt = None
    to_dt = None
    date_format = "%Y-%m-%d"

    if from_date:
        from_dt = datetime.strptime(from_date, date_format)
    if to_date:
        to_dt = datetime.strptime(to_date, date_format)

    session = requests.Session()

    while len(collected) < max_articles:
        params = {
            "limit": limit,
            "offset": offset,
            "fields": fields
        }
        if search_query:
            params["q"] = search_query

        try:
            response = session.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            items = data.get("items", [])
            total_available = data.get("total", 0)

            if not items:
                break

            for item in items:
                if len(collected) >= max_articles:
                    break
                headline = item.get("headline")
                url = item.get("url")
                snippet = item.get("subheadline")
                published_ts = item.get("published-at")
                this_author = item.get("author-name")

                published_dt = None
                if isinstance(published_ts, (int, float)):
                    published_dt = datetime.utcfromtimestamp(published_ts / 1000.0)
                else:
                    continue

                if author and this_author:
                    if author.lower() not in this_author.lower():
                        continue
                elif author and not this_author:
                    continue

                if from_dt and published_dt < from_dt:
                    continue
                if to_dt and published_dt > to_dt:
                    continue

                collected.append({
                    "title": headline,
                    "url": url,
                    "snippet": snippet,
                    "date": published_dt.isoformat() if published_dt else None,
                    "author": this_author,
                    "source": "New Indian Express"
                })

            offset += limit
            if offset >= total_available:
                break
        except Exception as e:
            print(f"[!] Error: {e}")
            break
    return parse_and_aggregate_articles(request, collected[:max_articles])


def translate_sentence_en_to_hi(request, sentence: str) -> str:
    """
    Translates a single English sentence into Hindi using the Helsinki-NLP model.
    """
    inputs = request.app.state.TRANSLATOR_TOKENIZER.encode(sentence, return_tensors="pt", truncation=True)
    outputs = request.app.state.TRANSLATOR.generate(inputs, max_length=512)
    hindi_translation = request.app.state.TRANSLATOR_TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    return hindi_translation


def translate_text_en_to_hi(request, english_text: str) -> str:
    """
    Splits the text into sentences, translates each one,
    and combines them into a single Hindi string.
    """
    # Split the large text into sentences
    sentences = request.app.state.NLTK.sent_tokenize(english_text)

    # Translate each sentence
    translated_sentences = []
    for sentence in sentences:
        hindi = translate_sentence_en_to_hi(request, sentence)
        translated_sentences.append(hindi)

    # Combine translated sentences back into one string
    final_hindi_text = " ".join(translated_sentences)
    return final_hindi_text


def determine_final_sentiment(sentiment_distribution):
    """
    Determine the final sentiment based on the following rules:
      - If Positive count is highest, return "Positive".
      - If Negative count is highest, return "Negative".
      - If Positive and Neutral are tied for highest, return "Positive".
      - If Negative and Neutral are tied for highest, return "Negative".
      - If Positive and Negative are tied for highest, return "Neutral".

    Args:
        sentiment_distribution (dict): Dictionary with keys "Positive", "Negative", "Neutral" and their counts.

    Returns:
        str: Final sentiment as per the rules.
    """
    pos = sentiment_distribution.get("Positive", 0)
    neg = sentiment_distribution.get("Negative", 0)
    neu = sentiment_distribution.get("Neutral", 0)
    max_val = max(pos, neg, neu)
    highest = [s for s, count in {"Positive": pos, "Negative": neg, "Neutral": neu}.items() if count == max_val]
    if len(highest) == 1:
        return highest[0]
    else:
        sentiment_set = set(highest)
        if sentiment_set == {"Positive", "Neutral"}:
            return "Positive"
        elif sentiment_set == {"Negative", "Neutral"}:
            return "Negative"
        elif sentiment_set == {"Positive", "Negative"}:
            return "Neutral"
        elif sentiment_set == {"Positive", "Negative", "Neutral"}:
            return "Neutral"


def convert_sets_to_lists(obj):
    """_summary_

    Args:
        obj (_type_): _description_

    Returns:
        _type_: _description_
    """
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: convert_sets_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_sets_to_lists(item) for item in obj]
    else:
        return obj