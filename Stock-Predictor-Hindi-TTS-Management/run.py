import os
os.environ["HOME"] = "/tmp"
os.environ["XDG_CACHE_HOME"] = "/tmp/.cache"
os.environ["MPLCONFIGDIR"] = "/tmp/.config/matplotlib"
os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache"

# Create the directories up front
for d in ("/tmp/.cache", "/tmp/.config/matplotlib", "/tmp/numba_cache", "/tmp/.local/share/tts_models"):
    os.makedirs(d, exist_ok=True)

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from pydantic import BaseModel
from TTS.api import TTS
app = FastAPI()

try:
    tts_model = TTS("tts_models/de/thorsten/tacotron2-DDC", progress_bar=False, gpu=False)
except Exception as e:
    raise RuntimeError(f"Failed to load TTS model: {e}")

class TTSRequest(BaseModel):
    text: str
    filename: str = "output.wav"


@app.post("/synthesize")
def synthesize_tts(request: TTSRequest):
    """
    Synthesize speech from Hindi text, stream the audio file to the API caller,
    and delete the file from the container after sending.
    """
    output_path = os.path.join("/tmp", request.filename)
    
    try:
        tts_model.tts_to_file(text=request.text, file_path=output_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during TTS synthesis: {e}")
    
    # Create a background task to remove the file after the response is sent
    cleanup_task = BackgroundTask(os.remove, output_path)
    
    return FileResponse(
        path=output_path,
        media_type="audio/wav",
        filename=request.filename,
        background=cleanup_task
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("run:app", host="0.0.0.0", port=5050, workers=1)
