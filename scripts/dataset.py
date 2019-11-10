import os
import json
import requests
import shutil

class instgram_data_set:
    def __init__(self,start_user,num_per_user,num_user=0,recraw=True,system='windows'):
        if recraw:
            if system=='windows':
                cmd = 'python ..\crawler\crawler.py posts_full -u ' + start_user + ' -n ' + str(
                    num_per_user)+' -o ..\crawler\output.json --fetch_hashtags'
                file_path='../crawler/output.json'
            elif system=='linux':
                cmd = 'python ../crawler/crawler.py posts_full -u ' + start_user + ' -n ' + str(
                    num_per_user) + ' -o ../crawler/output.json --fetch_hashtags'
                file_path='../crawler/output.json'
            else:
                print("OS should only be windows or linux")
                0/0
            os.system(cmd)

            with open('../crawler/output.json',errors='ignore', encoding='utf8') as json_file:
                posts = json.load(json_file)
            l=len(posts)
            for i in range(l):
                try:
                    posts[i]['hashtags'],posts[i]['img_urls']
                except:
                    continue
                hashtags=posts[i]['hashtags']
                j=0
                for url in posts[i]['img_urls']:
                    resp = requests.get(url, stream=True)
                    k=0
                    for hashtag in hashtags:
                        try:
                            local_file = open('../img/'+hashtag+'/'+str(i)+'-'+str(j)+'.jpg', 'wb')
                        except:
                            os.mkdir("../img/"+hashtag)
                            local_file = open('../img/' + hashtag + '/' + str(i) +'-' + str(j) + str(k) + '.jpg', 'wb')
                        resp.raw.decode_content = True
                        shutil.copyfileobj(resp.raw, local_file)
                        local_file.close()
                        k+=1
                    del resp
                    j+=1
                i+=1
            json_file.close()



object=instgram_data_set(start_user='juventus',num_per_user=100,recraw=False,system='windows')