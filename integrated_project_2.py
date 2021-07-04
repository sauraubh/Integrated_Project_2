#!/usr/bin/env python
# coding: utf-8

# # Project

# We work at a startup that sells food products and we need to investigate user behavior for the company's app. First, study the sales funnel to find out how users reach the purchase stage. Then, look at the results of an A/A/B test: The designers would like to change the fonts for the entire app, but the managers are afraid the users might find the new design intimidating. They decide to make a decision based on the results of an A/A/B test.

# In[1]:


get_ipython().system('pip install plotly --upgrade')


# In[2]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import math
import matplotlib.dates as mdates
import datetime 
import math
import plotly.express as px


# Imported all important libraries

# ## Open data and study information 

# In[3]:


data = pd.read_csv('/datasets/logs_exp_us.csv',sep='\t')
data.info()
data.head()


# Opened Data and looked for general information

# ### Data Preprocessing 

# In[4]:


data.columns = ['event_name', 'user_id', 'event_datetime', 'exp_id']


# We changed column names for better coding experiacne

# In[5]:


percentage = data.duplicated(keep=False).value_counts(normalize=True) * 100
print (percentage)


# we have 0.3 % of duplicates considering the amount of data and disoriented entries we can drop it

# In[6]:


data = data.drop_duplicates()


# we have dropped duplicates

# In[7]:


created_list= data.groupby('user_id', as_index=False).agg({'exp_id':'nunique'}).query('exp_id > 1')['user_id']
print(created_list)


# We don't have any duplicates here

# In[8]:


data['event_datetime']=pd.to_datetime(data['event_datetime'], unit='s')


# Here we have converted datetime from unix

# In[9]:


data['event_date'] = data['event_datetime'].dt.date
data['event_time'] = data['event_datetime'].dt.time
data.head()


# Added a separate date and time column

# ## Study and check the data 

# ### How many events are in the logs?How many users are in the logs?
#  

# In[10]:


tmp = data['event_name'].value_counts().rename_axis('event_name').reset_index(name='count')
tmp['event_name']


# There are 5 events in the logs: ['MainScreenAppear' 'OffersScreenAppear' 'CartScreenAppear'
#  'PaymentScreenSuccessful' 'Tutorial']

# In[11]:


len(data['user_id'].unique())


# There are 7551 unique users in the logs.

# ### What's the average number of events per user? 

# In[12]:


avgEvents = data.groupby('user_id')['event_name'].count().reset_index()
avgEvents.columns = ['user_id', 'num_of_events']
print(avgEvents['num_of_events'].mean())


# calculated the average number of events per user.The average number of events per user is 32.0

# ### What period of time does the data cover? Find the maximum and the minimum date. Plot a histogram by date and time. 

# In[13]:


uniqueEvents = data.groupby('user_id')['event_name'].nunique().reset_index().groupby('event_name')['user_id'].nunique().reset_index()
uniqueEvents.columns = ['event_number', 'unique_users']
uniqueEvents = uniqueEvents.sort_values(by='unique_users', ascending=False)
#uniqueEvents['event_name'] = data['event_name']
uniqueEvents


# The 7551 unique users in the logs are distributed by the event number as seen in the print out above. There were 2707 users who only completed event 1 , 1021 users who completed events 1 and 2, 317 users who completed events 1, 2, and 3  and 3035 users who went full circle by completing all events, including the payment screen successful. We expect every user to complete at least the 4 events and only 3035 / 7551 users actually complete all of those events.

# In[14]:


minDate = data['event_date'].min()
maxDate = data['event_date'].max()
print(minDate)
print(maxDate)


# determined min and max dates recorded.The period of time that the data covers is from: 2019-07-25 to 2019-08-07

# In[15]:


plt.figure(figsize=(15, 9))
plt.hist(data['event_datetime'],bins=10,color='green',alpha=0.7, rwidth=0.85)
plt.title('Date Frequencies')
plt.xlabel('Dates')
plt.ylabel('Number of Entries')
plt.show();


# The period of time that the data covers is from: 2019-07-25 to 2019-08-07. Once the frequency of the recorded dates are plotted, one can see that the majority of the date timestamps are from the first week of the month of August as there are over 30,000 entries for each day during that week. Although the period of time covers July 25th to August 8th of 2019, the moment where the data starts to be complete is on August 1st.

# In[16]:


timePeriod = data['event_date'].value_counts().rename_axis('event_date').reset_index(name='count').sort_values(by='event_date')


# In[17]:


plt.figure(figsize=(15, 9))
ax = sns.barplot(data = timePeriod, x='event_date', y='count')


plt.title('Date Frequencies')
plt.xlabel('Dates')
plt.ylabel('Number of Entries')


for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), 
                xy=(p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', 
                va = 'center', 
                xytext = (0, 10), 
                textcoords = 'offset points')
    

for item in ax.get_xticklabels():
    item.set_rotation(45)

plt.show()


# To understand conclusion above I tried also to plot Bar graph with dates and number of entries

# ### Find the moment at which the data starts to be complete and ignore the earlier section. What period does the data actually represent? 

# In[18]:


new_data = data[data['event_date'] >= pd.to_datetime('2019-08-01')]
new_data.head()


# removed older date entries

# ### Did you lose many events and users when excluding the older data? 

# In[19]:


oldnumOfusers = len(data['user_id'].unique())
oldnumOfevents = len(data['event_datetime'].unique())
print(oldnumOfusers)
print(oldnumOfevents)
newnumOfusers = len(new_data['user_id'].unique())
newnumOfevents = len(new_data['event_datetime'].unique())
print(newnumOfusers)
print(newnumOfevents)
print(oldnumOfusers-newnumOfusers)
print(oldnumOfevents-newnumOfevents)


# We have lost  17  users and  2610  events by removing any data entires from the month of July.
# It seems like before the August 1  there was a problem with data logging, because everything that comes before this date is inconsistent.  However, there were still users that registered through this period, moreover, some of them were active in the "normal" period. 

# In[20]:


percetn_change_user=((oldnumOfusers-newnumOfusers)/oldnumOfusers * 100)
percetn_change_event=((oldnumOfevents-newnumOfevents)/oldnumOfevents * 100)
print(percetn_change_user)
print(percetn_change_event)


# ### Make sure you have users from all three experimental groups.
#  

# In[21]:


expusers = new_data.groupby('exp_id')['user_id'].nunique().reset_index()
expusers.columns = ['exp_id', 'unique_users']
expusers


# We still have users presented from all 3 groups with 2484 users from group 246, 2513 users from group 247 and 2537 users from group 248.

# ## Study the event funnel 

# ### See what events are in the logs and their frequency of occurrence. Sort them by frequency. 

# In[22]:


eventsFreq = new_data['event_name'].value_counts().rename_axis('event_name').reset_index(name='count')
eventsFreq


# The result from value_counts() shows the frequency of each event in the logs: the most frequent event is the appearance of the main screen (117,431 events) which shrinks in number by more than half by the time the appearance of the offer sceen occurs (46,350 events). After the appearance of the offer sceen occurs, most of these events are brought to the appearance of the cart screen (42,365 events) and a large majority of these events are brought to payment screen successful (33,918 events). A substantially low number of events are the tutoral (1039 events).

# ### Find the number of users who performed each of these actions. Sort the events by the number of users. Calculate the proportion of users who performed the action at least once. 

# In[23]:


eventUsers = new_data.groupby('event_name')['user_id'].nunique().sort_values(ascending=False)
eventUsers = eventUsers.reset_index()
eventUsers.columns = ['event_name', 'user_unique']
eventUsers['user_percentage'] = (eventUsers['user_unique'] / new_data.user_id.nunique()) * 100


# found the number of users who performed each action

# In[24]:


at_least_once = new_data.groupby(['user_id', 'event_name'])['event_datetime'].count().reset_index()


# calculated the ratio of how many users performaed the action at least once. counted the number of event timestamps for each user for each event

# In[25]:


at_least_once = at_least_once[at_least_once['event_datetime'] > 1]


# extracted only the users who had more than one timestamp per event

# In[26]:


at_least_once = at_least_once.groupby('event_name')['user_id'].nunique() / new_data.groupby('event_name')['user_id'].nunique()
at_least_once = at_least_once.reset_index()
at_least_once.columns = ['event_name', 'at_least_once_percentage']
at_least_once['at_least_once_percentage'] = at_least_once['at_least_once_percentage'] * 100


# counted how many unique users had more than one action per event compared to all unique users

# In[27]:


eventUsers = eventUsers.merge(at_least_once, on='event_name')
eventUsers


# The number of users who performed each of these event actions can be seen in the table above: the user_unique column shows how many unique users performed each of the actions (any number of times) and the user_proportion column shows the percentage of users who performed each of these actions.
# 
# The proportion of users who performed the action at least once can be seen in the at_least_once_percentage column. As shown: out of the 7419 unique users that saw the main screen appear (98% of users), about 96% of them saw the main screen appear at least once. This column can be seen as a retention rate and tells us that a high percentage of users come back to the event.

# ### In what order do you think the actions took place. Are all of them part of a single sequence? You don't need to take them into account when calculating the funnel. 

# In[28]:


plt.figure(figsize=(11, 7))
ax = sns.barplot(data = eventUsers, x='event_name', y='user_unique')

plt.title('Users Performing Each Action')
plt.xlabel('Event Name')
plt.ylabel('Number of Uses')

for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), 
                xy=(p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', 
                va = 'center', 
                xytext = (0, 10), 
                textcoords = 'offset points')
    
for item in ax.get_xticklabels():
    item.set_rotation(45)

plt.show()


# The journey of users with each event name can be seen in the bar graph above where the most interacted with event is the appearance of the main screen with over 7000 users, which continues dropping as the sequence of events goes on.

# ### In what order do you think the actions took place? Are all of them part of a single sequence? 

# We believe that the order in which the actions took place is:
# 
# Main Screen -> Offer Screen -> Cart Screen -> Payment Successful Screen
# 
# They are not all part of a single sequence: it's possible to make a purchase without viewing the cart or make a purchase without seeing an offer pag.  not sure what the tutorial event is but  don't believe that it falls in a sequence.

# ### Use the event funnel to find the share of users that proceed from each stage to the next. At what stage do you lose the most users? What share of users make the entire journey from their first event to payment? 

# In[29]:


funnel = new_data.groupby('event_name')['user_id'].nunique().reset_index().sort_values(by='user_id', ascending=False)


# created funnel of unique users for each event

# In[30]:


funnel['percentage_change'] = funnel['user_id'].pct_change() * 100
funnel


# created percentage change from one funnel to another

# In[31]:


eventUsers = eventUsers.merge(funnel, on='event_name')
eventUsers =eventUsers.fillna(0)
eventUsers = eventUsers[['event_name', 'user_unique', 'user_percentage', 'percentage_change']]
eventUsers


# In the table above within the percentage_change column, we can see each event name, the number of unique users who had an action with th event and the percentage change from one event to antoher. All the users started out on the main screen but there was a 38% decrease in users who processed to the offer screen, etc. This is stage that lost the most users, with exception to the tutorial event which is not part of the logical sequence.
# 
# The share of users that make the entire journey from the first main screen appear event to the payment screen successful event can be seen in the user_percentage column in the payment screen successful row: about 47% of users complete all of the events.

# In[32]:


funnel_by_groups = []

for i in new_data.exp_id.unique():
    group = new_data[new_data.exp_id == i].groupby(['event_name', 'exp_id'])['user_id'].nunique().reset_index().sort_values(by='user_id', ascending=False)
    
    funnel_by_groups.append(group)

funnel_by_groups = pd.concat(funnel_by_groups)

fig = px.funnel(funnel_by_groups, x='user_id', y='event_name', color='exp_id')
fig.update_layout(title_text='funnel graph', title_x=0.5)
fig.show();


# plotted funnel visualization for better understanding of above explanation

# ## Study the results of the experiment 

# ### How many users are there in each group? 

# In[33]:


usersExp = new_data.groupby('exp_id')['user_id'].nunique().reset_index()
usersExp.columns = ['exp_id', 'unique_users']
usersExp


# The table above shows how many users there are in each group: there are about the same amount of users in each group, with group 246 being slightly smaller than the other groups.

# ### In each of the control groups, find the number of users who performed each event 

# In[34]:


expGroups = new_data.pivot_table(index='event_name', values='user_id', columns='exp_id', aggfunc=lambda x: x.nunique()).reset_index()
expGroups.columns = ['event_name', '246', '247', '248']
expGroups = expGroups.sort_values(by='246', ascending=False)
expGroups


# We can split up the user actions with event names by the different experiment groups, as shown by the table above. This shows the funnel according to experiment groups.

# ### In each of the control groups, find the number of users who performed the most popular event and find their share 

# In[35]:


mainGroups = expGroups[expGroups['event_name'] == 'MainScreenAppear']
mainGroups.columns = ['event_name', '246_performed', '247_performed', '248_performed']
mainGroups['246_share'] = mainGroups['246_performed'] / usersExp.loc[0,'unique_users'] * 100
mainGroups['247_share'] = mainGroups['247_performed'] / usersExp.loc[1,'unique_users'] * 100
mainGroups['248_share'] = mainGroups['248_performed'] / usersExp.loc[2,'unique_users'] * 100
mainGroups


# calculated share 
# The most popular event is the main screen appearing: the number of users who performed this event is shown under the performed columns and their share compared to the total can be seen in the share columns. For each group, over 98% of the users performed this main screen appear action.

# ### Check if there is a statistically significant difference between all of the control groups 

# Having 3 different experiment groups, it is important to ensure that that the results from these groups are based on fair numbers. In order to do so, we want to check if there is a statistically significant difference between all of the control groups. If we find that there is a significant difference, then the control groups have not be split up equally and any results we deduct will not accurately represent the population.

# In[36]:


expGroups = new_data.pivot_table(index='event_name', values='user_id', columns='exp_id', aggfunc=lambda x: x.nunique()).reset_index()
expGroups


# In[37]:


# find statistical significance for each group for each event
def check_hypothesis(group1, group2, alpha):

    # for every event
    for event in expGroups.event_name.unique():

        # define successes 
        successes1 = expGroups[expGroups.event_name == event][group1].iloc[0]
        successes2 = expGroups[expGroups.event_name == event][group2].iloc[0]

        # define trials
        trials1 = new_data[new_data.exp_id == group1]['user_id'].nunique()
        trials2 = new_data[new_data.exp_id == group2]['user_id'].nunique()

        # proportion for success in group 1
        p1 = successes1 / trials1

        # proportion for success in group 2
        p2 = successes2 / trials2

        # proportion in a combined dataset
        p_combined = (successes1 + successes2) / (trials1 + trials2)

        # define difference and z value
        difference = p1 - p2
        z_value = difference / math.sqrt(p_combined * (1 - p_combined) * (1/trials1 + 1/trials2))

        # calculate distribution
        distr = stats.norm(0,1)
        
         # calculate p_value
        p_value = (1 - distr.cdf(abs(z_value))) * 2
        print('p_value: ', p_value)
        if (p_value < alpha):
            print("Reject H0 for",event, 'and groups ',group1,' and ', group2, '\n')
        else:
            print("Fail to Reject H0 for", event,'and groups ',group1,' and ', group2, '\n')


# ### Calculate statistical difference between control groups 246 and 247 

# In[38]:


check_hypothesis(246, 247, 0.05)


# We want to test the statistical significance of the difference in conversion between control groups 246 and 247. This can be done using the CDF function which returns the expected probability for observing a value (number of unique users per event in group 246) less than or equal to a given value (number of unique users per event in group 247).
# 
# Null Hypothesis H0: There is no statistically significant difference in conversion between control groups 246 and 247. Alternative Hypothesis H1: There is a statistically significant difference in conversion between control groups 246 and 247.
# 
# For each event, the p_value is greater than the defined alpha level of 0.05 which means that we cannot reject the null hypothesis and we determine that there is a statistically significant difference between the two control groups for each event.

# ### Calculate statistical difference between control groups 246 and 248 

# In[39]:


check_hypothesis(246, 248, 0.05)


# We want to test the statistical significance of the difference in conversion between control groups 246 and 248. This can be done using the CDF function which returns the expected probability for observing a value (number of unique users per event in group 246) less than or equal to a given value (number of unique users per event in group 248).
# 
# Null Hypothesis H0: There is no statistically significant difference in conversion between control groups 246 and 248. Alternative Hypothesis H1: There is a statistically significant difference in conversion between control groups 246 and 248.
# 
# For each event, the p_value is greater than the defined alpha level of 0.05 which means that we cannot reject the null hypothesis and we determine that there is a statistically significant difference between the two control groups for each event.

# ### Calculate statistical difference between control groups 247 and 248 

# In[40]:


check_hypothesis(247, 248, 0.05)


# We want to test the statistical significance of the difference in conversion between control groups 247 and 248. This can be done using the CDF function which returns the expected probability for observing a value (number of unique users per event in group 247) less than or equal to a given value (number of unique users per event in group 248).
# 
# Null Hypothesis H0: There is no statistically significant difference in conversion between control groups 247 and 248. Alternative Hypothesis H1: There is a statistically significant difference in conversion between control groups 247 and 248.
# 
# For each event, the p_value is greater than the defined alpha level of 0.05 which means that we cannot reject the null hypothesis and we determine that there is a statistically significant difference between the two control groups for each event.

# ### Can you confirm that the groups were split properly? 

# After calculating the statistical differences between all control groups (246 and 247, 246 and 248, 247 and 247) with a CDF function, I found that the p_value was always greaters than the defined alpha level of 0.05 which means that there was a statistically significant difference between all the control groups for each event. Thus, it can be confirmed that the groups were not split properly.

# ### Calculate how many statistical hypothesis tests you carried out and run it through the Bonferroni correction. What should the significance level be? 

# In[41]:


alpha = 0.05 

# 2 control groups, 5 events per cotrol group
num_of_tests = 15
bonferroni_alpha = alpha/num_of_tests
fwer = 1 - (1 - bonferroni_alpha)**(num_of_tests)

print('The alpha level with Bonferroni correction is: ', bonferroni_alpha)
print('The probability of a type I error if we use an alpha value ', bonferroni_alpha, ' is: {:,.2%}'.format(fwer))


check_hypothesis(246, 247, 0.5)
check_hypothesis(246, 248, 0.5)
check_hypothesis(247, 248, 0.5)


# Previously, we set the alpha significance level to 0.05. Using the Bonferroni correction, we can use the family wise error rate formula to calculate a new alpha level which is dependent on the number of tests being performed. This error rate indicates the probability of making one or more false discoveries when performing multiple hypothesis tests. This error rate for our calculations comes out to be 0.5 which means that one in every five results could be false. Given this, we should change the significance level to be set to 0.5 to avoid an error rate of 20%.
# 
# When we ran the statistical significance tests again with an alpha level of 0.5: 7 out of the 15 tests were rejected which means that there is no statistical significance between more of the control groups than previously (where 0 out of the 15 tests were rejected.)

# ### Do the same thing for the group with altered fonts. Compare the results with those of each of the control groups for each event in isolation. Compare the results with the combined results for the control groups. What conclusions can you draw from the experiment? 

# In[42]:


new_data.loc[new_data['exp_id'] == 246, 'exp_id'] = 247
new_data


# Here we have changed group 246 to 247 (just name) so that we can calculate combined effect with group 246 and 247

# In[43]:


expGroups1 = new_data.pivot_table(index='event_name', values='user_id', columns='exp_id', aggfunc=lambda x: x.nunique()).reset_index()
expGroups1


# created new pivot_table to perform test

# In[44]:


# find statistical significance for each group for each event
def check_hypothesis(group1, group2, alpha):

    # for every event
    for event in expGroups1.event_name.unique():

        # define successes 
        successes1 = expGroups1[expGroups1.event_name == event][group1].iloc[0]
        successes2 = expGroups1[expGroups1.event_name == event][group2].iloc[0]

        # define trials
        trials1 = new_data[new_data.exp_id == group1]['user_id'].nunique()
        trials2 = new_data[new_data.exp_id == group2]['user_id'].nunique()

        # proportion for success in group 1
        p1 = successes1 / trials1

        # proportion for success in group 2
        p2 = successes2 / trials2

        # proportion in a combined dataset
        p_combined = (successes1 + successes2) / (trials1 + trials2)

        # define difference and z value
        difference = p1 - p2
        z_value = difference / math.sqrt(p_combined * (1 - p_combined) * (1/trials1 + 1/trials2))

        # calculate distribution
        distr = stats.norm(0,1)
        
         # calculate p_value
        p_value = (1 - distr.cdf(abs(z_value))) * 2
        print('p_value: ', p_value)
        if (p_value < alpha):
            print("Reject H0 for",event, 'and groups ',group1,' and ', group2, '\n')
        else:
            print("Fail to Reject H0 for", event,'and groups ',group1,' and ', group2, '\n')


# In[45]:


check_hypothesis(247, 248, 0.05)


# We want to test the statistical significance of the difference in conversion between control groups 247+246 and 248. This can be done using the CDF function which returns the expected probability for observing a value (number of unique users per event in group 247+246) less than or equal to a given value (number of unique users per event in group 248).
# 
# Null Hypothesis H0: There is no statistically significant difference in conversion between control groups 247+246 and 248. Alternative Hypothesis H1: There is a statistically significant difference in conversion between control groups 247+246 and 248.
# 
# For each event, the p_value is greater than the defined alpha level of 0.05 which means that we cannot reject the null hypothesis and we determine that there is a statistically significant difference between the two control groups for each event.

# In[46]:


alpha = 0.05 

# 2 control groups, 5 events per cotrol group
num_of_tests = 5
bonferroni_alpha = alpha/num_of_tests
fwer = 1 - (1 - bonferroni_alpha)**(num_of_tests)

print('The alpha level with Bonferroni correction is: ', bonferroni_alpha)
print('The probability of a type I error if we use an alpha value ', bonferroni_alpha, ' is: {:,.2%}'.format(fwer))

check_hypothesis(247, 248, 0.5)


# Previously, we set the alpha significance level to 0.05. Using the Bonferroni correction, we can use the family wise error rate formula to calculate a new alpha level which is dependent on the number of tests being performed. This error rate indicates the probability of making one or more false discoveries when performing multiple hypothesis tests. This error rate for our calculations comes out to be 0.5 which means that one in every five results could be false. Given this, we should change the significance level to be set to 0.5 to avoid an error rate of 20%.
# 
# When we ran the statistical significance tests again with an alpha level of 0.5: 3 out of the 5 tests were rejected which means that there is no statistical significance between more of the control groups than previously (where 0 out of the 5 tests were rejected.)

# # Final Analysis

# 1. Aim: To analyse behaviour of customers for food product startup
# 2. There are Five different event with 7551 users with 32 average users per event
# 3. There are 4 events when user completes it we consider it as complete use of events where the number of users who completed all 4 events are 3035 which is less than 50%
# 4. We got consistent data after 01.08.2019 over 30000 entries for each day in week
# 5. Out of the 7419 unique users that saw the main screen appear (98% of users), about 96% of them saw the main screen appear at least once. This column can be seen as a retention rate and tells us that a high percentage of users come back to the event.
# 6. order in which the actions took place is:Main Screen -> Offer Screen -> Cart Screen -> Payment Successful Screen
# 7. There was a 38% decrease in users who processed to the offer screen, etc. This is stage that lost the most users, with exception to the tutorial event which is not part of the logical sequence.
# 8. about 47% of users complete all of the events
# 9. There are about the same amount of users in each group, with group 246 being slightly smaller than the other groups
# 10. over 98% of the users performed this main screen appear action
# 11. The Test showed there is statastical difference in control groups when we performed 3 control groups with 5 eventsUsing the Bonferroni correction, we can use the family wise error rate formula to calculate a new alpha level which is dependent on the number of tests being performed. per group so, 15 test in total
# 12. Hypothesis was decided on each comparison of control groups as follows: Null Hypothesis H0: There is no statistically significant difference in conversion between control groups. Alternative Hypothesis H1: There is a statistically significant difference in conversion between control groups
# 13. p_value was always greaters than the defined alpha level of 0.05 which means that there was a statistically significant difference between all the control groups for each event. Thus, it can be confirmed that the groups were not split properly
# 14. Using the Bonferroni correction, we can use the family wise error rate formula to calculate a new alpha level which is dependent on the number of tests being performed
# 15. statistical significance tests again with an alpha level of 0.5: 7 out of the 15 tests were rejected which means that there is no statistical significance between more of the control groups than previously (where 0 out of the 15 tests were rejected.)
# 16. When Compared the results with the combined results for the control groups there was no change it was similar conclusion mentioned in point 14 & 15

# We want to test the statistical significance of the difference in conversion between control groups 246,248 and 247.
# expected probability for observing a value (number of unique users per event in group 247,246) less than or equal to a given value (number of unique users per event in group 248) and Vice Versa.
# We found out that there is a statistically significant difference between the two control groups for each event
# The significant differences between the A groups, this can help us uncover factors that may be distorting the results. 
# significant difference shows group not splitted properly.
# significant difference caused the control groups have not be split up equally and any results we deduct will not accurately represent the population.
# 
