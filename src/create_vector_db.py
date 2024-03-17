import click
from qdrant_client import QdrantClient, models
from openai import OpenAI
from tqdm import tqdm
import json
import requests
import os
from prompts import REVIEWS_SYSTEM_PROMPT, REVIEWS_USER_PROMPT

TRIPADVISOR_API_KEY = os.environ.get('TRIPADVISOR_API_KEY')


def save_json(data, path):
    with open(path, "w") as outfile:
        json.dump(data, outfile)


def get_df(dataset_path, is_hf):
    if is_hf:
        from datasets import load_dataset
        dataset = load_dataset(dataset_path)
        return dataset['train'].to_pandas()
    else:
        import pandas as pd
        return pd.read_csv(dataset_path)


def _concat_reviews(df):
    text = ''
    for _, row in df.iterrows():
        text += '\n'
        if row.review_title:
            text += '\nTitle:\n' + row.review_title
        if row.review_text:
            text += '\nReview:\n' + row.review_text

    return text


def create_reviews_symmary(df, model, hotels, pos_rate=4.0, neg_rate=4.0, n_reviews=6):
    """Create a summary of reviews for each hotel, based on the most positive and most negative reviews.

    Args:
        df (pd.DataFrame): hotels dataset
        model (str): OpenAI model name
        hotels (list): list of hotels to create summaries for
        pos_rate (float): minimum positive rate, inclusive
        neg_rate (float): maximum negative rate, exclusive
        n_reviews (int): number of reviews to consider for each category

    Returns:
        dict: hotel name -> reviews summary
    """
    df['review_text_len'] = df.review_text.str.len().fillna(value=0)
    df['review_title_len'] = df.review_title.str.len().fillna(value=0)

    client = OpenAI()
    hotels_reviews_summary = {}
    for hotel in tqdm(hotels):
        temp = df[df.hotel_name.eq(hotel)]
        temp_pos = temp[temp.rate >= pos_rate].nlargest(n_reviews, 'review_text_len')
        temp_neg = temp[temp.rate < neg_rate].nlargest(n_reviews, 'review_text_len')
        if len(temp_pos) == 0 and len(temp_neg) == 0:
            temp_pos = temp.nlargest(n_reviews, 'review_title_len')

        text = _concat_reviews(temp_pos) + _concat_reviews(temp_neg)

        if text:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": REVIEWS_SYSTEM_PROMPT},
                    {"role": "user", "content": REVIEWS_USER_PROMPT.format(text=text)},
                ]
            )
            hotels_reviews_summary[hotel] = response.choices[0].message.content
        return hotels_reviews_summary


def _get_loc_id(hotel):
    """In order to get the hotel info, we need to get the location id first.

    Args:
        hotel (str): hotel name

    Returns:
        str: location id
    """
    url = "https://api.content.tripadvisor.com/api/v1/location/search?key={key}&searchQuery={hotel}&category=hotels&language=en"
    headers = {"accept": "application/json"}

    response = requests.get(url.format(hotel=hotel, key=TRIPADVISOR_API_KEY), headers=headers)
    try:
        return response.json()['data'][0]['location_id']
    except Exception as e:
        print(f'{response.status_code=}')
        print(f'{response.text=}')
        print(f'Error: {e}')
        return None


def get_hotel_info(hotel):
    """Get hotel info from TripAdvisor.
        Following information is retrieved from the TripAdvisor API:
            - rank
            - ratings distributions
            - subratings
            - amenities

    Args:
        hotel (str): hotel name

    Returns:
        dict: hotel info
    """
    url = "https://api.content.tripadvisor.com/api/v1/location/{loc_id}/details?key={key}&language=en&currency=USD"
    headers = {"accept": "application/json"}

    loc_id = _get_loc_id(hotel)
    if loc_id is None:
        return None
    response = requests.get(url.format(loc_id=loc_id, key=TRIPADVISOR_API_KEY), headers=headers)
    try:
        response = response.json()
    except Exception as e:
        print(f'{response.status_code=}')
        print(f'{response.text=}')
        print(f'Error: {e}')
        return None
    rank = response['ranking_data'].get('ranking_string')
    reviews_ratings = response.get('review_rating_count')
    subratings = {}
    for d in response['subratings']:
        subratings[response['subratings'][d]['name']] = response['subratings'][d]['value']
    amenities = response.get('amenities', [])
    return dict(
        rank=rank,
        reviews_ratings=reviews_ratings,
        subratings=subratings,
        amenities=amenities,
    )


def get_desc(hotel, data):
    """Create a description of the hotel based on the retrieved data from TripAdvisor.

    Args:
        hotel (str): hotel name
        data (dict): hotel info
    
    Returns:
        str: hotel description
    """
    rating = "Rating: "+str(data[hotel]['rank'])+". "

    distr_ranks = "Rating distribution "
    for key in data[hotel]['reviews_ratings'].keys():
        distr_ranks += str(key) + ": " + str(data[hotel]['reviews_ratings'][key] + ", ")
    distr_ranks = distr_ranks[:-2]+". "

    sub_ranks = "Specific ratings: "
    if 'rate_location' in data[hotel]['subratings'].keys():
        sub_ranks += "Location " + data[hotel]['subratings']['rate_location'] + ", "

    if 'rate_sleep' in data[hotel]['subratings'].keys():
        sub_ranks += "Sleep " + data[hotel]['subratings']['rate_sleep'] + ", "
    if 'rate_room' in data[hotel]['subratings'].keys():
        sub_ranks += "Room " + data[hotel]['subratings']['rate_room'] + ", "
    if 'rate_service' in data[hotel]['subratings'].keys():
        sub_ranks += "Service " + data[hotel]['subratings']['rate_service'] + ", "
    if 'rate_cleanliness' in data[hotel]['subratings'].keys():
        sub_ranks += "Cleanliness " + data[hotel]['subratings']['rate_cleanliness']
    sub_ranks += ". "

    amenities = "Amenities available: "
    for i in data[hotel]['amenities']:
        amenities += str(i) + ", "
    amenities = amenities[:-2] + "."

    total_desc = rating + distr_ranks + sub_ranks + amenities
    return total_desc


def get_payload(hotel, df):
    """Create a metadata which will be collected in the database.

    Args:
        hotel (str): hotel name
        df (pd.DataFrame): hotels dataset

    Returns:
        dict: metadata
    """
    temp = df[df.hotel_name.eq(hotel)]
    rating = temp.rating_value.value_counts().index[0]
    city = temp.locality.value_counts().index[0]
    country = temp.country.value_counts().index[0]
    price = temp.price_range.str.split(' ').str[0].value_counts().index[0]
    return dict(
        hotel_name=hotel,
        rating=rating,
        city=city,
        country=country,
        price=price
    )


@click.command()
@click.option('--dataset-path', default='traversaal-ai-hackathon/hotel_datasets', help='Path to the dataset.')
@click.option('--is-hf', is_flag=True, default=True, help='Whether the dataset is in huggingface format, csv otherwise.')
@click.option('--db-path', default='data/db', help='Path to the output database.')
@click.option('--collection-name', default='hotels', help='Name of the collection in the database.')
@click.option('--embeddings-model', default='text-embedding-3-large', help='Name of the model to use for embeddings.')
@click.option('--embeddings-size', default=3072, help='Size of the embeddings.')
@click.option('--reviews-model', default='gpt-3.5-turbo-0125', help='Name of the model to use for reviews summary.')
def create_vector_db(dataset_path, is_hf, db_path, collection_name, embeddings_model, embeddings_size, reviews_model):
    REVIEW_SUMMARIES_PATH = 'reviews_summary.json'
    HOTELS_INFO_PATH = 'hotels_info.json'

    df = get_df(dataset_path, is_hf)

    # Create a collection if it does not exist and filter out hotels that are already in the collection
    qdrant_client = QdrantClient(path=db_path)
    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=embeddings_size, distance=models.Distance.COSINE),
        )
        hotels = df.hotel_name.unique()
    else:
        docs, _ = qdrant_client.scroll(
            collection_name=collection_name,
            limit=1e9,
            with_payload=True,
            with_vectors=False,
        )
        hotels = set(df.hotel_name.unique()) - set([doc.payload['hotel_name'] for doc in docs])
    if len(hotels) == 0:
        return

    # Create reviews summary using OpenAI
    reviews_summary = create_reviews_symmary(df, reviews_model, hotels)
    save_json(reviews_summary, REVIEW_SUMMARIES_PATH)

    # Get hotel info from TripAdvisor
    hotels_info = {}
    for hotel in tqdm(hotels):
        hotels_info[hotel] = get_hotel_info(hotel)
    save_json(hotels_info, HOTELS_INFO_PATH)

    # Create descriptions and payloads for each hotel
    texts = []
    payloads = []
    for hotel in hotels:
        trip_desc_hotel = get_desc(hotel, hotels_info)
        review_hotel = reviews_summary.get(hotel)
        payload = get_payload(hotel, df)
        text = trip_desc_hotel if trip_desc_hotel else ''  + '\n' + review_hotel if review_hotel else ''
        payload['description'] = text
        payloads.append(payload)
        texts.append(text)

    # Create description embeddings and upsert them to the database
    openai_client = OpenAI()
    embeddings = openai_client.embeddings.create(input=texts, model=embeddings_model)
    points = [
        models.PointStruct(
            id=idx,
            vector=data.embedding,
            payload=payload,
        )
        for idx, (data, payload) in enumerate(zip(embeddings.data, payloads))
    ]
    qdrant_client.upsert(collection_name, points)


if __name__ == '__main__':
    create_vector_db()
