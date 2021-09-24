#!/usr/bin/env python
# coding: utf-8

# # World Problems
# 
# The purpose of this task is to distribute a given number of works amongst several machines. Each machine has an efficiency - time (measured in minutes) to complete each given work. Moreover, each machine requires some time - break between works (in minutes). Data/table is provided as a CSV file in the attachments “WorkDistribution.csv”. 
# <br><br>
# Your goal is to minimise the total time to complete all works.
# <br><br>
# Formulate this as a linear programming problem, solve it by applying different methods and analyse solutions obtained. Discuss and compare the performance of the methods applied.

# In[1]:


# import pulp and pandas
from pulp import *
import pandas as pd


# ## Setting up data
# Read in the csv file provided, taking the number of rows as the number of machines, and the number of columns minus 2 (for first column machine headings and last row break times) as the number of work items
# <br><br>
# Then assign the data in the csv file to a 2D array to be referenced later and the break times each machine requires

# In[2]:


# read in csv file with data
df = pd.read_csv('WorkDistribution.csv')

# define number of work items and machines to be distributed on
n_work = (len(df.columns) - 2)
n_machines = len(df)

# create an empty 2d array
time_taken = [[0 for x in range(n_work)] for y in range(n_machines)]

# assign data from dataframe to 2d array
for i in range(n_machines):
    for j in range(n_work):
        time_taken[i][j] = df.iloc[i,j+1]

# set up empty break time array for each machine
m_break = [0 for x in range(n_machines)]

# assign break data from data frame to array
for i in range(n_machines):
    m_break[i] = df.iloc[i, len(df.columns) - 1]


# ## Setting Up LP Problem with PuLP
# First define the problem as a minimisation problem. Then set up binary variables to determine what work piece is done on what machine, how many breaks are required for each machine and if the machine is used to process work.

# In[3]:


# Define the model as minimisation problem
op_work_prob = LpProblem("Optimal_Work_Distribution", LpMinimize)


# In[4]:


# set up binary variables for work piece i being done on machine j
work_1 = LpVariable.dicts('work_1', range(1, n_machines + 1),cat='Binary')
work_2 = LpVariable.dicts('work_2', range(1, n_machines + 1),cat='Binary')
work_3 = LpVariable.dicts('work_3', range(1, n_machines + 1),cat='Binary')
work_4 = LpVariable.dicts('work_4', range(1, n_machines + 1),cat='Binary')
work_5 = LpVariable.dicts('work_5', range(1, n_machines + 1),cat='Binary')
work_6 = LpVariable.dicts('work_6', range(1, n_machines + 1),cat='Binary')
work_7 = LpVariable.dicts('work_7', range(1, n_machines + 1),cat='Binary')
work_8 = LpVariable.dicts('work_8', range(1, n_machines + 1),cat='Binary')
work_9 = LpVariable.dicts('work_9', range(1, n_machines + 1),cat='Binary')
work_10 = LpVariable.dicts('work_10', range(1, n_machines + 1),cat='Binary')
work_11 = LpVariable.dicts('work_11', range(1, n_machines + 1),cat='Binary')
work_12 = LpVariable.dicts('work_12', range(1, n_machines + 1),cat='Binary')
work_13 = LpVariable.dicts('work_13', range(1, n_machines + 1),cat='Binary')
work_14 = LpVariable.dicts('work_14', range(1, n_machines + 1),cat='Binary')
work_15 = LpVariable.dicts('work_15', range(1, n_machines + 1),cat='Binary')

# put all dictionaries in an array
work = [work_1, work_2, work_3, work_4, work_5, work_6, work_7, work_8
     , work_9, work_10, work_11, work_12, work_13, work_14, work_15]


# In[5]:


# Set up binary variabes to determine if a break needs to be added
m1_break = LpVariable.dicts('m1_break', range(1, n_work + 1),cat='Binary')
m2_break = LpVariable.dicts('m2_break', range(1, n_work + 1),cat='Binary')
m3_break = LpVariable.dicts('m3_break', range(1, n_work + 1),cat='Binary')
m4_break = LpVariable.dicts('m4_break', range(1, n_work + 1),cat='Binary')
m5_break = LpVariable.dicts('m5_break', range(1, n_work + 1),cat='Binary')
m6_break = LpVariable.dicts('m6_break', range(1, n_work + 1),cat='Binary')
m7_break = LpVariable.dicts('m7_break', range(1, n_work + 1),cat='Binary')
m8_break = LpVariable.dicts('m8_break', range(1, n_work + 1),cat='Binary')
m9_break = LpVariable.dicts('m9_break', range(1, n_work + 1),cat='Binary')

# put all breaks in an array
breaks = [m1_break, m2_break, m3_break, m4_break, m5_break, m6_break, m7_break, m8_break, m9_break]


# In[6]:


# Set up binary variabes to determine what machines need to be used
m1_used = LpVariable('m1_used', cat='Binary')
m2_used = LpVariable('m2_used', cat='Binary')
m3_used = LpVariable('m3_used', cat='Binary')
m4_used = LpVariable('m4_used', cat='Binary')
m5_used = LpVariable('m5_used', cat='Binary')
m6_used = LpVariable('m6_used', cat='Binary')
m7_used = LpVariable('m7_used', cat='Binary')
m8_used = LpVariable('m8_used', cat='Binary')
m9_used = LpVariable('m9_used', cat='Binary')


# ### Objective function
# Set up objective function with the sum of each work piece time taken to process by the machine it is assigned to, plus the sum of the break time required for each machine.

# In[7]:


# set up objective function looping through all work pieces on all machines
# and if the binary value is one multiply it by the time taken to do that work item on that machine
# Then loop through each break variable and if the binary value is one 
# multiply it by the break required on that machine
op_work_prob += (lpSum([(work[i][j + 1] * (time_taken[j][i])) for i in range (n_work) for j in range(n_machines)]) +
         lpSum([(breaks[i][j + 1] * (m_break[i])) for i in range (n_machines) for j in range(n_work)])
        ), "Objective Function"


# ### Constraints
# First set of constraints is ensuring each work item is only processed on one machine

# In[8]:


# declare constraints that each work piece is completed once
op_work_prob += lpSum([work_1[i] for i in range(1, n_machines + 1)]) == 1, "work piece 1 done once"
op_work_prob += lpSum([work_2[i] for i in range(1, n_machines + 1)]) == 1, "work piece 2 done once"
op_work_prob += lpSum([work_3[i] for i in range(1, n_machines + 1)]) == 1, "work piece 3 done once"
op_work_prob += lpSum([work_4[i] for i in range(1, n_machines + 1)]) == 1, "work piece 4 done once"
op_work_prob += lpSum([work_5[i] for i in range(1, n_machines + 1)]) == 1, "work piece 5 done once"
op_work_prob += lpSum([work_6[i] for i in range(1, n_machines + 1)]) == 1, "work piece 6 done once"
op_work_prob += lpSum([work_7[i] for i in range(1, n_machines + 1)]) == 1, "work piece 7 done once"
op_work_prob += lpSum([work_8[i] for i in range(1, n_machines + 1)]) == 1, "work piece 8 done once"
op_work_prob += lpSum([work_9[i] for i in range(1, n_machines + 1)]) == 1, "work piece 9 done once"
op_work_prob += lpSum([work_10[i] for i in range(1, n_machines + 1)]) == 1, "work piece 10 done once"
op_work_prob += lpSum([work_11[i] for i in range(1, n_machines + 1)]) == 1, "work piece 11 done once"
op_work_prob += lpSum([work_12[i] for i in range(1, n_machines + 1)]) == 1, "work piece 12 done once"
op_work_prob += lpSum([work_13[i] for i in range(1, n_machines + 1)]) == 1, "work piece 13 done once"
op_work_prob += lpSum([work_14[i] for i in range(1, n_machines + 1)]) == 1, "work piece 14 done once"
op_work_prob += lpSum([work_15[i] for i in range(1, n_machines + 1)]) == 1, "work piece 15 done once"


# Second set of constraints determine if each machine is used, this is used to determine how many breaks are required for each machine in the next set of constraints. 

# In[9]:


# delcare constraints to set mi_used binary to 1 if work is processed on machine i, 0 otherwise
op_work_prob += lpSum([work[i][1] for i in range(n_work)]) >= m1_used, "if machine 1 if used"
op_work_prob += lpSum([work[i][2] for i in range(n_work)]) >= m2_used, "if machine 2 if used"
op_work_prob += lpSum([work[i][3] for i in range(n_work)]) >= m3_used, "if machine 3 if used"
op_work_prob += lpSum([work[i][4] for i in range(n_work)]) >= m4_used, "if machine 4 if used"
op_work_prob += lpSum([work[i][5] for i in range(n_work)]) >= m5_used, "if machine 5 if used"
op_work_prob += lpSum([work[i][6] for i in range(n_work)]) >= m6_used, "if machine 6 if used"
op_work_prob += lpSum([work[i][7] for i in range(n_work)]) >= m7_used, "if machine 7 if used"
op_work_prob += lpSum([work[i][8] for i in range(n_work)]) >= m8_used, "if machine 8 if used"
op_work_prob += lpSum([work[i][9] for i in range(n_work)]) >= m9_used, "if machine 9 if used"


# The third set of constraints determine how many breaks each machine requires. If a machine is required to process 4 work items, it requires 3 breaks, one in between each work item but not at the end. This is as simple as saying the number of breaks required is one less than the number of work pieces processed on that machine. 
# <br><br>
# However, this forces work on every machine as if a machine is required to process no work, the breaks equates to minus 1, which is not possible as the breaks are binary (0 or 1). This is where the binary variable for if the machine is used is required. If the machine is used (i.e. binary value is 1) it is multiplied by 1 and added to the number of breaks, equalling the number of jobs ran. If the machine is not used (value is 0), the zero is multiplied by 1, adding zero to the number of breaks. In this way, the number of breaks can equal zero then there is no work processed on the machine.

# In[10]:


# declare constraints that the number of breaks on each machine is one less 
# than the number of work items carried out on that machine if that machine is used
op_work_prob += lpSum([breaks[0][i] for i in range(1, n_work + 1)]) + (m1_used) == lpSum([work[i][1] for i in range(n_work)]), "machine 1 breaks required"
op_work_prob += lpSum([breaks[1][i] for i in range(1, n_work + 1)]) + (m2_used) == lpSum([work[i][2] for i in range(n_work)]), "machine 2 breaks required"
op_work_prob += lpSum([breaks[2][i] for i in range(1, n_work + 1)]) + (m3_used) == lpSum([work[i][3] for i in range(n_work)]), "machine 3 breaks required"
op_work_prob += lpSum([breaks[3][i] for i in range(1, n_work + 1)]) + (m4_used) == lpSum([work[i][4] for i in range(n_work)]), "machine 4 breaks required"
op_work_prob += lpSum([breaks[4][i] for i in range(1, n_work + 1)]) + (m5_used) == lpSum([work[i][5] for i in range(n_work)]), "machine 5 breaks required"
op_work_prob += lpSum([breaks[5][i] for i in range(1, n_work + 1)]) + (m6_used) == lpSum([work[i][6] for i in range(n_work)]), "machine 6 breaks required"
op_work_prob += lpSum([breaks[6][i] for i in range(1, n_work + 1)]) + (m7_used) == lpSum([work[i][7] for i in range(n_work)]), "machine 7 breaks required"
op_work_prob += lpSum([breaks[7][i] for i in range(1, n_work + 1)]) + (m8_used) == lpSum([work[i][8] for i in range(n_work)]), "machine 8 breaks required"
op_work_prob += lpSum([breaks[8][i] for i in range(1, n_work + 1)]) + (m9_used) == lpSum([work[i][9] for i in range(n_work)]), "machine 9 breaks required"


# ## Solve the LP Problem
# The problem is then solved, with each binary value equalling 1 being displayed. From this we can determine which work piece is run on each machine.

# In[11]:


# solve problem and store status
status = op_work_prob.solve()

# print the status
print(f"Solution: {LpStatus[status]}\n")

# print instructions of how to read the results
print("Read mi_break_x as 1 break for machine i. mi_break_x appearing twice means machine i requires two breaks.")
print("Read mi_used machine i is required to process work.")
print("Read work_i_j as work piece i is to be proccessed by machine j.")
print("Example: work_10_5 means work piece 10 is run on machine 5.\n")

# Show the variables that are 1 (i.e. the machine work piece j is run on and the breaks included)
for v in op_work_prob.variables():
    if(v.value() == 1):
        print(f"{v.name}: {v.value()}")

# print the mimimum time taken
print(f"\nMinimum time value (in minutes): {op_work_prob.objective.value()}")

