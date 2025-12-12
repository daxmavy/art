import numpy as np
from dataclasses import dataclass
import random
import math
import shortuuid
import pandas as pd
import shutil
import os

def get_alpha_beta_from_mean_variance(mu, sigma):
    alpha = ((1-mu)/sigma**2 - 1/mu) * (mu)**2
    alpha = np.clip(a= alpha, a_min = 0.001, a_max=None)
    beta = alpha * (1/mu - 1)
    if np.any(alpha < 0):
        raise Exception(f'alpha < 0, alpha: \n{alpha} \n beta: \n {beta} \n mu \n {mu} \n sigma \n {sigma}')
    # if alpha < 0:
    #     print('mu, sigma:', mu, sigma)
    #     print('alpha, beta: ', alpha, beta)
    return alpha, beta

def score_diff_to_probability(diff): 
    return 1 / (1 + math.e**(diff))

def normalise_vector(vec):
    return (vec / np.sum(vec))

@dataclass
class Artist():
    id : str
    # avg_characteristicness : float # not strictly bounded, but should sit in [0, 1]
    std_characteristicness : float 
    artist_characteristicness_aspects: np.ndarray # should be in [0,1]
    artist_aspect_weighting: np.ndarray

    def __post_init__(self):
        self.artist_aspect_weighting = np.array(self.artist_aspect_weighting)
        if np.any(self.artist_aspect_weighting < 0):
            raise Exception(f'weighting is non-negative {self.artist_aspect_weighting}')
        
        self.artist_aspect_weighting = normalise_vector(self.artist_aspect_weighting)

    def create_real_ArtImage(self, subject):

        alphas, betas = get_alpha_beta_from_mean_variance(self.artist_characteristicness_aspects, self.std_characteristicness)

        jittered_aspects = np.random.beta(alphas, betas)

        return ArtImage(
            id = f'img~{shortuuid.uuid()}',
            is_ai = False,
            artist = self,
            creator = self,
            subject = subject,
            characteristicness_aspects = jittered_aspects
        )
    
    def to_dict(self):
        return {
            'id': self.id,
            'style_aspects': self.artist_characteristicness_aspects,
            'std_characteristicness': self.std_characteristicness,
            'artist_aspect_weighting': self.artist_aspect_weighting
        }
    
@dataclass
class ArtImage():
    id: str
    is_ai: bool # whether the ArtImage is AI or real
    # characteristicness: float # between 0 and 1
    artist: Artist
    creator: Artist
    subject: str
    characteristicness_aspects: np.ndarray

    def to_dict(self):
        return {
            'id': self.id,
            'is_ai': self.is_ai,
            'artist': self.artist.id,
            'creator': self.creator.id,
            'subject': self.subject,
            'style_aspects': self.characteristicness_aspects
        }

@dataclass
class AIArtImage(ArtImage):
    base_real_image: ArtImage

    def to_dict(self):
        di = super().to_dict()
        di['base_image'] = self.base_real_image.id
        return di

@dataclass
class Participant():
    id: str
    skill: float # average value of 5 is reasonable (see calibration section below)
    participant_aspect_weighting: np.ndarray # this should sum to 1; forced below

    def __post_init__(self):
        if np.any(self.participant_aspect_weighting < 0):
            raise Exception(f'weighting is non-negative {self.participant_aspect_weighting}')
        
        self.participant_aspect_weighting = normalise_vector(self.participant_aspect_weighting)

    def compare_two_ArtImages(self, ArtImage1 : ArtImage, ArtImage2 : ArtImage):
        comparison_weighting_vector = normalise_vector(self.participant_aspect_weighting + ArtImage1.artist.artist_aspect_weighting)
        
        img1_char = np.sum(np.multiply(
            comparison_weighting_vector,
            ArtImage1.characteristicness_aspects
        ))

        img2_char = np.sum(np.multiply(
            comparison_weighting_vector,
            ArtImage2.characteristicness_aspects
        ))

        diff = img2_char - img1_char
        prob = score_diff_to_probability(self.skill * diff)
        
        diffs = self.skill * np.multiply(
                ArtImage1.characteristicness_aspects - ArtImage2.characteristicness_aspects,
                comparison_weighting_vector
        )

        if random.uniform(0, 1) <= prob:
            chosen =  ArtImage1
            reason = np.argmax(diffs)
        else:
            chosen = ArtImage2
            reason = np.argmin(diffs)
        
        return chosen, reason
    
    def to_dict(self):
        return {
            'id': self.id,
            'skill': self.skill,
            'style_aspect_weights': self.participant_aspect_weighting
        }
        
        
@dataclass
class ImageModel():
    id : str
    ability_to_mimic_style_aspects: np.ndarray # [0,1]
    # this should only rarely be higher than an artist's own characteristicness

    def __post_init__(self):
        self.ability_to_mimic_style_aspects = np.array(self.ability_to_mimic_style_aspects)

    
    def create_ai_ArtImage_matching_image(self, image: ArtImage, std_characteristicness: float) -> ArtImage:
        new_aspects = np.multiply(
            self.ability_to_mimic_style_aspects,
            image.characteristicness_aspects
        )

        alphas, betas = get_alpha_beta_from_mean_variance(new_aspects, std_characteristicness)

        jittered_aspects = np.random.beta(alphas, betas)

        return AIArtImage(
            id = f'img~{shortuuid.uuid()}',
            is_ai = True,
            artist = image.artist,
            creator = self, # type: ignore
            subject = image.subject,
            characteristicness_aspects=jittered_aspects,
            base_real_image = image
        )

    def to_dict(self):
        return {
            'id': self.id,
            'style_mimicry_abilities': self.ability_to_mimic_style_aspects
        }

def generate_data_for_simulation(
        folder_name = None,
        std_in_participant_aspect_weighting_vectors = 0.02,
        num_participants = 40,
        avg_participant_skill = 5,
        std_participant_skill = 0.5,
        global_latent_aspect_weighting = np.ones(shape=4),
        ARTIST_ASPECT_VECTORS = [
            0.95*np.ones(shape=4)
        ]*4,
        ARTIST_ASPECT_WEIGHTING = [
            np.ones(shape=4)
        ]*4,
        std_real_artist_characteristicness = 0.05,
        Model_ABILITIES = [
            0.8 * np.ones(shape=4)
        ]*3,
        num_subjects_per_artist = 2,
        model_names = [f'model_{i}' for i in range(3)],
        artist_names = [f'artist_{i}' for i in range(4)]
):
    if folder_name is not None:
        shutil.rmtree(folder_name, ignore_errors=True)
        os.mkdir(folder_name)

    global_latent_aspect_weighting = normalise_vector(global_latent_aspect_weighting)

    artists = [ Artist(
        # id=f'art~{shortuuid.uuid()}',
        id = name,
        artist_characteristicness_aspects = vec,
        std_characteristicness=std_real_artist_characteristicness,
        artist_aspect_weighting=weighting
    ) for name, vec, weighting in zip(artist_names, ARTIST_ASPECT_VECTORS, ARTIST_ASPECT_WEIGHTING) ]

    real_artworks = []
    for artist in artists:
        for subject in range(num_subjects_per_artist):
            real_artworks.append(artist.create_real_ArtImage(subject=subject))

    models = [ ImageModel(
        # id = shortuuid.uuid(),
        id = name,
        ability_to_mimic_style_aspects = vec
    ) for name, vec in zip(model_names, Model_ABILITIES)]

    ai_artworks = []
    for real_artwork in real_artworks:
        for Model in models:
            ai_artworks.append(Model.create_ai_ArtImage_matching_image(real_artwork, std_real_artist_characteristicness))

    comparisons = [(ai_artwork.base_real_image, ai_artwork) for ai_artwork in ai_artworks]
    
    skills = np.random.normal(
        loc = avg_participant_skill, 
        scale = std_participant_skill, 
        size=num_participants
    )

    aspect_weighting_alphas, aspect_weighting_betas = get_alpha_beta_from_mean_variance(
        global_latent_aspect_weighting, 
        std_in_participant_aspect_weighting_vectors
    )
    
    aspect_weightings = np.random.beta(
        a = aspect_weighting_alphas, 
        b = aspect_weighting_betas, 
        size = (num_participants, len(global_latent_aspect_weighting))
    )

    participants = [
        Participant(
            f'par~{shortuuid.uuid()}', 
            skill = s,
            participant_aspect_weighting=vec
        )
        for s, vec in zip(skills, aspect_weightings)
    ]

    trial_outcomes  = []
    for participant in participants:
        for real_artwork, ai_artwork  in comparisons:
            chosen_ArtImage, reason = participant.compare_two_ArtImages(real_artwork, ai_artwork)
            record = {
                'participant' : participant.id,
                'real_artwork' : real_artwork.id,
                'ai_artwork' : ai_artwork.id,
                'artist' : real_artwork.artist.id,
                'subject' : real_artwork.subject,
                'Model' : ai_artwork.creator.id,
                'real_win': int(chosen_ArtImage == real_artwork),
                'ai_win' : int(chosen_ArtImage != real_artwork),
                'winner': chosen_ArtImage.id,
                'loser': [el for el in [real_artwork.id, ai_artwork.id] if el != chosen_ArtImage.id][0],
                'reason': reason
            }
            trial_outcomes.append(record)

    part_df = pd.DataFrame([el.to_dict() for el in participants])
    artist_df = pd.DataFrame([el.to_dict() for el in artists])
    artwork_df = pd.DataFrame([el.to_dict() for el in real_artworks + ai_artworks])
    model_df = pd.DataFrame([el.to_dict() for el in models])
    outcome_df = pd.DataFrame(trial_outcomes)
    
    if folder_name is not None:
        part_df.to_csv(f'{folder_name}/participants.csv', index=False)
        artist_df.to_csv(f'{folder_name}/artists.csv', index=False)
        artwork_df.to_csv(f'{folder_name}/artworks.csv', index=False)
        model_df.to_csv(f'{folder_name}/models.csv', index=False)
        outcome_df.to_csv(f'{folder_name}/outcomes.csv', index=False)

    return {
        'outcomes': outcome_df, 
        'models': model_df, 
        'artworks': artwork_df,
        'artists': artist_df, 
        'participants': part_df
    }
