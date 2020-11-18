# Python Document Detector
A simple document detector in python. 

# Test run
## Document Extractor
```
from ocr.page_extractor import PageExtractor

extractor = PageExtractor()
doc = extractor(PATH_TO_FILE)
```
## OCR CNIC
```
from ocr.ocr_document import OCRProcessor

processor = OCRProcessor()
ocr_json = processor(PATH_TO_FILE)
```
