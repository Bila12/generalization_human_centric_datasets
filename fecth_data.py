import tmdbsimple as tmdb
tmdb.API_KEY = 
image_path = 'https://image.tmdb.org/t/p/w600_and_h900_bestv2'
import requests
import pandas as pd
import datetime
movie_dat = pd.DataFrame( columns=['id', 'title', 'genre','Plot', 'rating', 'vote_count', 'release_date'])
count = 0
text_file = “list_clear_id_train”
with open(text_file, "rb") as fp:
    file_id = pickle.load(fp)
for i in file_id:
    try:
        movie = tmdb.Movies(i)
        response = movie.info()
        ppath = movie.poster_path
    
        path = image_path + ppath
        image_name = './posters_AC/poster_AC_' + str(i) + '.jpg'
        genre = movie.genres
        title = movie.title.replace('\n', ' ')
        plot = movie.overview.replace('\n', ' ')
        idm = i
        rating = movie.vote_average
        vote_count = movie.vote_count
        date = movie.release_date
        
        print(date)
        movie_dat.loc[count] = [idm, title, genre, plot, rating, vote_count, date]
        img_data = requests.get(path).content
        with open(image_name, 'wb') as handler:
            handler.write(img_data)
        count = count +1
    except:
        continue
        
movie_dat.to_csv('./posters/movie_data_rating', encoding='utf-8', index=False)
