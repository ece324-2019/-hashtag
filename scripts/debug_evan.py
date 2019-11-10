import os
import json
from pprint import pprint
from ekphrasis.classes.segmenter import Segmenter
from ekphrasis.classes.segmenter import Segmenter
seg = Segmenter(corpus="twitter")
print("ok")
# loading data from the json file
data = []
with open('../crawler/output.json') as json_data:
    data = json.load(json_data)
    json_data.close()
    pprint(data)


# tokenize the hashtags

hashtagLists = ["instagood", "#ThrowBackThursday", "forThegram", "#GodIsGreat", "#CheeseParole"]
for item in hashtagLists:
    print(seg.segment(item))
