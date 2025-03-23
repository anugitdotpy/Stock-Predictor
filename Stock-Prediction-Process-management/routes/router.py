import os
import json
import traceback
from fastapi import FastAPI, APIRouter, Request, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel
from typing import Optional
from src import (
    scrape_indian_express_api_fil,
    generate_comparative_analysis_llm,
    translate_text_en_to_hi,
    determine_final_sentiment,
    convert_sets_to_lists
)


routes_router = APIRouter()


class ArticleQuery(BaseModel):
    search_query: Optional[str] = None
    author: Optional[str] = None
    from_date: Optional[str] = None
    to_date: Optional[str] = None
    max_articles: Optional[int] = 10


@routes_router.api_route("/process-management/process-input", methods=["POST"])
async def process_input(request: Request, query: ArticleQuery):
    """
    Expects a JSON payload with parameters such as search_query, author, from_date, to_date,
    and max_articles. It calls the scraping function and returns the response as a JSON file.
    """
    try:
        response_data = scrape_indian_express_api_fil(
            request,
            search_query=query.search_query,
            author=query.author,
            from_date=query.from_date,
            to_date=query.to_date,
            max_articles=query.max_articles
        )
        print("RESPONSE DATA: \n %s \n", response_data)
        try:
            response_data[
                "comparative sentiment score"
            ]["Coverage Differences"] = generate_comparative_analysis_llm(
                request,
                response_data["articles"]
            ).split('\n\n', 1)[1]
        except:
            response_data[
                "comparative sentiment score"
            ]["Coverage Differences"] = generate_comparative_analysis_llm(
                request,
                response_data["articles"]
            )
        
        print("GENERATED DATA: \n %s \n", response_data)
        output_filename = "news_summary.json"
        output_path = os.path.join("/tmp", output_filename)
        hindi_text = translate_text_en_to_hi(
            request,
            response_data["comparative sentiment score"]["Coverage Differences"]
        )
        print("HINDI DATA: \n %s \n", hindi_text)
        response_data["hindi_translation"] = hindi_text
        final_sentiment = determine_final_sentiment(
            response_data["comparative sentiment score"]["sentiment distribution"]
        )
        if final_sentiment == "Positive":
            response_data["Final Sentiment Analysis"] = f"{query.search_query}'s overall news coverage is positive. Potential stock growth is expected"
        elif final_sentiment == "Negative":
            response_data["Final Sentiment Analysis"] = f"{query.search_query}'s overall news coverage is negative. Potential stock decline is expected"
        else:
            response_data["Final Sentiment Analysis"] = f"{query.search_query}'s overall news coverage is negative. Less chance for any large variation in stock statistics"
        # Write the response data to the file as JSON
        converted_response = convert_sets_to_lists(response_data)
        with open(output_path, "w") as f:
            json.dump({"response": converted_response}, f)
        # Define a background task to delete the file after it is sent
        cleanup_task = BackgroundTask(os.remove, output_path)
        return FileResponse(
            path=output_path,
            media_type="application/json",
            filename=output_filename,
            background=cleanup_task
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error fetching articles.")