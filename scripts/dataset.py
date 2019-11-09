import os
import json

class instgram_data_set:
    def __init__(self,start_user,num_per_user,num_user,retrieve=True):
        os.system("python ../crawler/crawler.py post_full -u "+start_user+" -n "+str(num_per_user)+" -o ../crawler/output.jason --fetch_hashtags")
        with open("../crawler/output.json") as json_file:
            posts = json.load(json_file)
        l=len(posts)
        for i in range(l):
            try:
                posts[i]={'key':posts[i]['key'],'img_urls':posts[i]['image_urls'],'hashtags':posts[i]['hashtags']}
            except:
                pass
        pass



object=instgram_data_set("juventus",100,0)