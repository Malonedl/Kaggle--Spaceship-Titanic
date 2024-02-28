import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
Getting set up
"""
dataTest = pd.read_csv("test.csv")
dataTrain = pd.read_csv("train.csv")

full = pd.concat([dataTrain, dataTest], ignore_index=True)
df = pd.DataFrame(full)

print("test", dataTest)
print("train", dataTrain)
print("full", full)
print(full.info)

"""
Extracting Families
"""


"""
Extracting Group and Group number from passenger ID
"""
# Group and subgroup are seperated by an '_'
df[['Group', 'Subgroup']] = df['PassengerId'].str.split('_', expand=True)
print(df)

"""
Checking to see if it would be reasonable to create a "traveling partners" feature using group, homeplant, and destination.
"""
# get the counts of the people who have the same group, homeplanet and destination
group_counts = df.groupby(['Group', 'HomePlanet', 'Destination']).size().reset_index(name='Count')
print("Group Counts:\n", group_counts)

# counts of people in the same group
total_group_counts = df.groupby('Group').size().reset_index(name='TotalCount')

# merge the counts back together
groups_df = pd.merge(group_counts, total_group_counts, on='Group')

# calculate the likelihood
groups_df['%AllTogether'] = groups_df['Count'] / groups_df['TotalCount'] * 100

print("\nLikelihood:\n", groups_df)

group_travel_percent = groups_df['%AllTogether'].sum() / (groups_df['%AllTogether'].count())
print("Percent of groups that started together and ended together:", group_travel_percent)
# 88% of people in the same group are starting together and ending together.

"""
Converting bools into ints
"""

"""
Separating Cabin into Deck/num/side and remove cabin
"""


