from pydantic import BaseModel
from cat.mad_hatter.decorators import plugin


class MySettings(BaseModel):
    search_max_results: int = 3

@plugin
def settings_model():
    return MySettings