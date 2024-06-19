from fastapi import APIRouter, HTTPException
from app.crawler import getArticle
from app.model import sentiment_model

router = APIRouter()

@router.get("/analyze_sentiment")
async def analyze_sentiment(coin: str):
    if coin:
        try:
            data = await getArticle(coin)
            titles = data["titles"]
            count = data["count"]
            res = '\n'.join(titles)

            predicted_label = sentiment_model.predict_sentiment(res)

            sentiment_classes = {0: 'Neutral', 1: 'Positive', 2: 'Negative'}
            sentiment = sentiment_classes[predicted_label]

            return {
                "coin": coin,
                "sentiment": sentiment,
                "article_count": count,
                "articles": titles
            }
        except Exception as e:
            import traceback
            traceback_str = ''.join(traceback.format_tb(e.__traceback__))
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}\n{traceback_str}")
    else:
        raise HTTPException(status_code=400, detail="Please provide a valid coin name")
