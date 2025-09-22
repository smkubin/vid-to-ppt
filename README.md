\# vid-to-ppt

This is a tool to translate trainigs/lections distributed as videos of presentation slides plus speaker's voice. 

It converts it back to slides, and each slide is annotated with what the speaker says when the slide is displayed.

Written mostly by chatgpt.



\# usage

pip install -r requirements.txt



python vid-to-ppt.py <mpeg\_file>





\# fine-tuning

SLIDE\_INTERVAL: float = 1.0     # how often do we capture video frames. usually 1 sec works, 2-3 sec is faster but in corner cases can miss some slides.



SIMILARITY: float = 0.85        # similarity threshold (0–1) - increase to detect smaller changes.



WHISPER\_MODEL: str = "base"     # whisper model size (tiny, base, small, medium, large)



