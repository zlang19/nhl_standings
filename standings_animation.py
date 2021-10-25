#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import datetime

from bs4 import BeautifulSoup
import requests
from PIL import Image
from matplotlib import colors

import plotly
import plotly.graph_objects as go
import plotly.express as px


# In[2]:


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


# In[3]:


todays_date = datetime.date.today()


# In[4]:


# Variable to store game results
game_result_list = []

# Scrape hockey reference for game results
url = 'https://www.hockey-reference.com/leagues/NHL_2021_games.html'
html_doc = requests.get(url).text
soup = BeautifulSoup(html_doc, 'html.parser')

# Iterate through each row representing a game
for table_row in soup.find('table', {'id': 'games'}).find('tbody').find_all('tr'):
    if 'class' in table_row:
        continue

    # Parse date
    date_string = table_row.find('th').get_text()
    date = datetime.datetime.strptime(date_string, '%Y-%m-%d').date()

    # Parse game details
    parsed_table_row = [x.get_text() for x in table_row.find_all('td')]
    away_team = parsed_table_row[0]
    away_score = None if parsed_table_row[1] == '' else int(parsed_table_row[1])
    home_team = parsed_table_row[2]
    home_score = None if parsed_table_row[3] == '' else int(parsed_table_row[3])

    # Parse overtime
    overtime = parsed_table_row[4]
    went_overtime = (overtime != '')
    overtime_string = f' in {overtime}' if went_overtime else ''

    # Leave results blank if game hasn't been played yet
    if date >= todays_date:
        away_game_points = np.nan
        home_game_points = np.nan
        away_team_description = None
        home_team_description = None
    # If game has been played calculate points and description
    else:
        away_game_points = 2 * int(away_score > home_score) + 1 * int(away_score < home_score and went_overtime)
        home_game_points = 2 * int(home_score > away_score) + 1 * int(home_score < away_score and went_overtime)

        # Format description for away team
        away_team_win = away_score > home_score
        away_team_description = 'Beat the ' if away_team_win else 'Lost to the '
        away_team_description += home_team
        away_team_description += f' {away_score}-{home_score}'
        away_team_description += overtime_string

        # Format description for home team
        home_team_description = 'Lost to the ' if away_team_win else 'Beat the '
        home_team_description += away_team
        home_team_description += f' {home_score}-{away_score}'
        home_team_description += overtime_string

    # Store game from each teams perspective
    game_result_list.append((date, away_team, home_team, away_game_points, away_team_description))
    game_result_list.append((date, home_team, away_team, home_game_points, home_team_description))

# Load games into dataframe and calulcate games played and total points
game_result_df = pd.DataFrame(game_result_list, columns=['Date', 'Team Name', 'Other Team Name', 'Game Points', 'Description'])
game_result_df['Games Played'] = game_result_df.groupby('Team Name').cumcount() + 1
game_result_df['Total Points'] = game_result_df.groupby('Team Name')['Game Points'].cumsum()


# In[5]:


# Variable to store season results and prediction at a specified date
game_result_df_at_date_list = []

# Start at beginning of the season and itererate through to today
season_start_date = game_result_df['Date'].min()
for days_since_season_start in range((todays_date - season_start_date).days):
    evaluation_date = season_start_date + datetime.timedelta(days=days_since_season_start)
    print(evaluation_date)

    # Copy game results
    game_result_df_at_date = game_result_df.copy()

    # Remove known game results for games that come after evaluation date
    game_result_df_at_date['Predicted'] = game_result_df_at_date['Date'] >= evaluation_date
    game_result_df_at_date.loc[game_result_df_at_date['Predicted'], 'Game Points'] = np.nan
    game_result_df_at_date.loc[game_result_df_at_date['Predicted'], 'Description'] = None
    game_result_df_at_date.loc[game_result_df_at_date['Predicted'], 'Total Points'] = np.nan

    # Calculate expected results for each matchup between teams. Calculated by average ppg against a given team
    known_game_result_df_at_date = game_result_df_at_date[~game_result_df_at_date['Predicted']]
    expected_results_df = known_game_result_df_at_date.groupby(['Team Name', 'Other Team Name']).apply(lambda matchup_df: round(matchup_df['Game Points'].sum() / matchup_df.shape[0], 3))

    # Iterate through each matchup with a calculated expected result
    for matchup, expected_result in expected_results_df.iteritems():
        # Fill in expected results
        unknown_matchup_index = (game_result_df_at_date[['Team Name', 'Other Team Name']] == matchup).all(axis=1) & game_result_df_at_date['Game Points'].isna()
        game_result_df_at_date.loc[unknown_matchup_index, 'Game Points'] = expected_result
        game_result_df_at_date.loc[unknown_matchup_index, 'Description'] = f'Predicted {expected_result} point(s) against {matchup[1]}'

    # Fill in missing game points and descriptions for unknown matchups
    # If a matchup doesn't exist, best guess is a tie. 1 point for each team
    game_result_df_at_date['Game Points'] = game_result_df_at_date['Game Points'].fillna(1)
    game_result_df_at_date['Description'] = game_result_df_at_date.apply(lambda row: row['Description'] if row['Description'] else f'Predicted {1} point(s) against {row["Other Team Name"]}', axis=1)
    game_result_df_at_date['Total Points'] = game_result_df_at_date.groupby('Team Name')['Game Points'].cumsum()
    game_result_df_at_date['Points Above Average'] = game_result_df_at_date.groupby(['Games Played']).apply(lambda x: x['Total Points'] - x['Total Points'].mean()).droplevel(0).round(3)

    # Set evaulation date and store
    game_result_df_at_date['Evaluation Date'] = evaluation_date.strftime('%Y-%m-%d')
    game_result_df_at_date_list.append(game_result_df_at_date)


# In[6]:


# Construct final dataframe used for plotting
df = pd.concat(game_result_df_at_date_list)

# Create a starting point for each team line, 0 games and 0 points
season_start_df = df.groupby(['Team Name', 'Evaluation Date']).first().reset_index()
season_start_df['Date'] = season_start_date - datetime.timedelta(days=1)
season_start_df['Other Team Name'] = ''
season_start_df['Game Points'] = 0
season_start_df['Description'] = ''
season_start_df['Games Played'] = 0
season_start_df['Total Points'] = 0
season_start_df['Predicted'] = False
season_start_df['Points Above Average'] = 0

# Create starting point for each prediction line, picking up where the known results end
prediction_start_df = df[~df['Predicted']].groupby(['Team Name', 'Evaluation Date']).last().reset_index()
prediction_start_df['Predicted'] = True

# Sort and reset index
df = pd.concat([df, season_start_df, prediction_start_df]).sort_values(['Evaluation Date', 'Date', 'Team Name'], kind='mergesort').reset_index(drop=True)

# Final y value will be points above average plus a little offset incase some teams scores overlap
# df['y'] = df['Points Above Average'] - df.groupby(['Division', 'Games Played', 'Points Above Average']).cumcount() / 15
df['y'] = df['Points Above Average'] - df.groupby(['Games Played', 'Points Above Average', 'Evaluation Date']).cumcount() / 15


# Define order of which to show plots
# division_order = ['All', 'West', 'Central', 'East', 'North']

# Line name will be the team name + weather or not is a predicted score
# df['line_id'] = df['Team Name'] + df['Predicted'].map(lambda x: f' Predicted' if x else '')

# def grey_out_hex(hex_color):
#     hex_color = colors.to_hex(colors.to_rgb(hex_color))
#     return '#' + \
#         hex(round((int(hex_color[1:3], 16) + 255) / 2))[2:] + \
#         hex(round((int(hex_color[3:5], 16) + 255) / 2))[2:] + \
#         hex(round((int(hex_color[5:7], 16) + 255) / 2))[2:]

# color_discrete_map = {}
# for team_name, team_color in team_colors.iteritems():
#     color_discrete_map[team_name] = team_color
#     color_discrete_map[f'{team_name} Predicted'] = grey_out_hex(team_color)


# In[7]:


# Create Figure
fig = px.line(
    df, x='Games Played', y='y', 
    color='Team Name', line_dash='Predicted', color_discrete_map=team_colors.to_dict(), 
    hover_data={
        'Team Name':True, 'Games Played':True, 'Points Above Average':True, 'Description':True, 
        'y':False, #'line_id':False,
        # 'Division':False
    },
    # width=1200, height=2400, 
    width=1200, height=600, 
    range_x=[0, df['Games Played'].max()], range_y=[df[~df['Predicted']]['Points Above Average'].min()-2, df[~df['Predicted']]['Points Above Average'].max()+2], 
    # facet_row='Division', category_orders={'Division':division_order}, 
    animation_frame='Evaluation Date', 
)

# Highlight 0
fig.add_hline(y=0)
# fig.update_layout(showlegend=False)
fig.update_layout(title=f'NHL Team Standings {datetime.date.today().isoformat()}', legend={'title':'Team Name'})
fig.update_xaxes(dtick=1)
fig.update_yaxes(title='Points Above Average', dtick=2)

# Group known and predicted lines into same legend key
for frame_i in range(len(fig.frames)):
    for data_i in range(len(fig.frames[frame_i].data)):
        go_name, is_predicted = fig.frames[frame_i].data[data_i]['name'].split(', ')
        fig.frames[frame_i].data[data_i]['name'] = go_name
        fig.frames[frame_i].data[data_i]['legendgroup'] = go_name
        if is_predicted == 'True':
            fig.frames[frame_i].data[data_i]['showlegend'] = False

print('Done creating figure')


# In[8]:


# Get final location of each team on the plot so that we can put their logo there
# final_locations = df.groupby(['Team Name', 'Division']).last().reset_index()[['Team Name', 'Division', 'Games Played', 'Points Above Average']]
final_locations = df[~df['Predicted']].groupby(['Team Name', 'Evaluation Date']).last().reset_index()[['Team Name', 'Evaluation Date', 'Games Played', 'Points Above Average']]

# Create offsets for teams that end up stacking
# final_locations['Offset'] = final_locations.groupby(['Division', 'Games Played', 'Points Above Average']).cumcount() / 3
final_locations['Offset'] = final_locations.groupby(['Games Played', 'Points Above Average', 'Evaluation Date']).cumcount() / 3

# Get ordered list of evaluation dates
evaluation_date_list = list(df['Evaluation Date'].unique())

# Create variable to store images
layout_image_list_list = [[] for i in range(len(evaluation_date_list))]

# Iterate through each final location of a team for each frame
# for team_name, division, games_played, score, offset in final_locations.values:
for team_name, evaluation_date, games_played, score, offset in final_locations.values:
    # Caluclate position including any offsets
    img_x = games_played + offset
    img_y = score

    # Locate logo and load in the image. Need to load in prior to plotting or else it wont show up in the html file
    logos_path = './logos'
    logo_src = f'{logos_path}/{team_name}.png'
    logo = Image.open(logo_src)

    # Store image
    # fig.add_layout_image(
    layout_image = go.layout.Image(
        source=logo,
        # row=division_order[::-1].index(division)+1, col=1,
        xref="x", yref="y",
        # layer='below',
        x=img_x, y=img_y,
        xanchor="center", yanchor="middle",
        sizex=2, sizey=2,
        opacity=0.8,
    )

    layout_image_list_list[evaluation_date_list.index(evaluation_date)].append(layout_image)

# Iterate through each frame and add images to the layout
for frame_i, layout_image_list in enumerate(layout_image_list_list):
    fig.frames[frame_i]['layout'] = go.Layout(images=layout_image_list)

print('Done adding images')


# In[19]:


# Write fig to file
fig.write_html('animation_index.html')
print('Done writing to file')


# In[ ]:




