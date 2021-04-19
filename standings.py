#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import datetime

from bs4 import BeautifulSoup
import requests
from PIL import Image

import plotly
import plotly.graph_objects as go
import plotly.express as px


# In[2]:


start = datetime.date(2021, 1, 13)
end = datetime.date.today()
date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]

game_result_list = []

for curr_date in date_generated:
    url = f'https://www.hockey-reference.com/boxscores/?year={curr_date.year}&month={curr_date.month}&day={curr_date.day}'
    html_doc = requests.get(url).text
    soup = BeautifulSoup(html_doc, 'html.parser')

    game_summaries = soup.find("div", {"class": "game_summaries"})

    for game_summary in game_summaries.find_all("div", {"class": "game_summary"}):
        parsed_game_summary = [x.get_text() for x in game_summary.find_all('td')]
        away_team, away_score, _, home_team, home_score, went_ot = parsed_game_summary
        away_score = int(away_score)
        home_score = int(home_score)
        went_ot = (went_ot[:2] == 'OT' or went_ot[:2] == 'SO')

        game_result_list.append((away_team, away_score, home_team, home_score, went_ot))

game_result_df = pd.DataFrame(game_result_list, columns=['Away Team', 'Away Score', 'Home Team', 'Home Score', 'Went OT'])

game_result_df


# In[3]:


game_team_names = game_result_df[['Away Team', 'Home Team']].values.flatten()
team_names, game_counts = np.unique(game_team_names, return_counts=True)

# Variables to store results in
game_result_str_list = []
point_df = pd.DataFrame(np.nan, index=pd.Index(team_names, name='Team Name'), columns=pd.Index(range(game_counts.max() + 1), name='Games Played'))
point_df[0] = 0

for team_name in team_names:
    games_played = 0
    team_points = 0
    for _, game_result in game_result_df.iterrows():
        if game_result['Away Team'] == team_name:
            team_location = 'Away'
            other_team_location = 'Home'
        elif game_result['Home Team'] == team_name:
            team_location = 'Home'
            other_team_location = 'Away'
        else:
            continue

        games_played += 1
        team_score = game_result[f'{team_location} Score']
        other_team = game_result[f'{other_team_location} Team']
        other_team_score = game_result[f'{other_team_location} Score']

        # Update game result string list
        game_result_str = 'Beat the ' if team_score > other_team_score else 'Lost to the '
        game_result_str += other_team
        game_result_str += f' {team_score}-{other_team_score}'
        went_ot_str = ' in OT' if game_result['Went OT'] else ''
        game_result_str += went_ot_str
        game_result_str_list.append((team_name, games_played, game_result_str))

        # Update points dataframe
        if team_score > other_team_score:
            team_points += 2
        elif game_result['Went OT']:
            team_points += 1
        point_df.at[team_name, games_played] = team_points

game_result_str_series = pd.DataFrame(game_result_str_list, columns=['Team Name', 'Games Played', 'Game Result String']).set_index(['Team Name', 'Games Played'])['Game Result String']

point_df


# In[4]:


# Get map from division to team
division_to_teams = {
    'Central': ['Carolina Hurricanes', 'Chicago Blackhawks', 'Columbus Blue Jackets', 'Dallas Stars', 'Detroit Red Wings', 'Florida Panthers', 'Nashville Predators', 'Tampa Bay Lightning', ],
    'East': ['Boston Bruins', 'Buffalo Sabres', 'New Jersey Devils', 'New York Islanders', 'New York Rangers', 'Philadelphia Flyers', 'Pittsburgh Penguins', 'Washington Capitals', ],
    'West': ['Anaheim Ducks', 'Arizona Coyotes', 'Colorado Avalanche', 'Los Angeles Kings', 'Minnesota Wild', 'San Jose Sharks', 'St. Louis Blues', 'Vegas Golden Knights', ],
    'North': ['Calgary Flames', 'Edmonton Oilers', 'Montreal Canadiens', 'Ottawa Senators', 'Toronto Maple Leafs', 'Vancouver Canucks', 'Winnipeg Jets']
}

# Get map from team to division
team_to_division = {}
_ = [team_to_division.update({team:division for team in team_list}) for division, team_list in division_to_teams.items()]

# Read in team colors
team_colors = pd.read_csv('logos/team_colors.csv', index_col=0, squeeze=True)


# In[5]:


# Normalize points by subtracting average
# Forward fill points so that teams that have more games played dont set the average
# Do this for divisions
division_point_diff_df = point_df.groupby(team_to_division).apply(lambda division_point_df: division_point_df - division_point_df.ffill(axis=1).mean(axis=0))
division_point_diff_df.index = pd.MultiIndex.from_tuples(division_point_diff_df.index.map(lambda team_name: (team_name, team_to_division[team_name])), names=['Team Name', 'Division'])
# Do this for whole NHL
all_point_diff_df = point_df - point_df.ffill(axis=1).mean(axis=0)
all_point_diff_df.index = pd.MultiIndex.from_product([all_point_diff_df.index, ['All']], names=['Team Name', 'Division'])
all_point_diff_df
# Combine into single point difference dataframe
point_diff_df = pd.concat([all_point_diff_df, division_point_diff_df])
# Stack values so that plotly can break up each team into their own line. Round 3 decimals and Fix naming
df = point_diff_df.stack()
df = (df * 1000).astype(int) / 1000
df = df.reset_index().rename(columns={0:'Points Above Average'})
# Final y value will be points above average plus a little offset incase some teams scores overlap
df['y'] = df['Points Above Average'] - df.groupby(['Division', 'Games Played', 'Points Above Average']).cumcount() / 15
# Get description of each game
df['Game'] = df.apply(lambda row: '' if row['Games Played'] == 0 else game_result_str_series[(row['Team Name'], row['Games Played'])], axis=1)

# Define order of which to show plots
division_order = ['All', 'West', 'Central', 'East', 'North']

# Create Figure
fig = px.line(df, x='Games Played', y='y', 
color='Team Name', color_discrete_map=team_colors.to_dict(), 
hover_data={'Team Name':True, 'Games Played':True, 'Points Above Average':True, 'Game':True, 'y':False, 'Division':False},
width=1200, height=2400, range_x=[0, point_df.shape[1]], range_y=[df['Points Above Average'].min()-2, df['Points Above Average'].max()+2], 
facet_row='Division', category_orders={'Division':division_order})

# Highlight 0
fig.add_hline(y=0)
# fig.update_layout(showlegend=False)
fig.update_layout(title=f'NHL Team Standings {datetime.date.today().isoformat()}')
fig.update_xaxes(dtick=1)
fig.update_yaxes(title='Points Above Average', dtick=2)

# Get final location of each team on the plot
final_locations = df.groupby(['Team Name', 'Division']).last().reset_index()[['Team Name', 'Division', 'Games Played', 'Points Above Average']]
# Create offsets for those that stack
final_locations['Offset'] = final_locations.groupby(['Division', 'Games Played', 'Points Above Average']).cumcount() / 3
for team_name, division, games_played, score, offset in final_locations.values:
    img_x = games_played + offset
    img_y = score

    # Locate logo and load in the image. Need to load in prior to plotting or else it wont show up in the html file
    logos_path = './logos'
    logo_src = f'{logos_path}/{team_name}.png'
    logo = Image.open(logo_src)

    # Plot image
    fig.add_layout_image(
        source=logo,
        row=division_order[::-1].index(division)+1, col=1,
        xref="x", yref="y",
        layer='below',
        x=img_x, y=img_y,
        xanchor="center", yanchor="middle",
        sizex=2, sizey=2,
        opacity=0.8,
    )

# print('done')
fig.write_html('index.html')
fig.show()


# In[6]:


def point_streak(row):
    longest_streak = 0
    curr_streak = 0
    prev_x =  0
    for x in row.values[1:]:
        if x <= prev_x+1:
            curr_streak += 1
        else:
            curr_streak = 0
        if curr_streak > longest_streak:
            longest_streak = curr_streak
        prev_x = x
    return longest_streak

point_df.apply(point_streak, axis=1).sort_values(ascending=False)


# In[ ]:




