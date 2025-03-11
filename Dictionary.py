#1-sorted dict by key
my_dict = {'b': 2, 'a': 1, 'c': 3, 'd': 0}
sorted_dict = dict(sorted(my_dict.items()))  # Sorts by keys
print(sorted_dict)  # Output: {'a': 1, 'b': 2, 'c': 3, 'd': 0}

#2-Sort Python Dictionary by Key or Value
my_dict = {'a': 3, 'b': 1, 'c': 4, 'd': 2}
sorted_items = sorted(my_dict.items(), key=lambda item: item[1])
print(sorted_items)  # Output: [('b', 1), ('d', 2), ('a', 3), ('c', 4)]

#3- Handling Missing keys in Dictionaries
country_code = {'India' : '0091','Australia' : '0025','Nepal' : '00977'}
 # search dictionary for country code of India
print(country_code.get('India', 'Not Found'))   #0091

#4- find the sum of all items in a dictionary
d = {'a': 100, 'b': 200, 'c': 300}
res = sum(d.values())
print(res)
res = sum([d[key] for key in d])
print(res)

#5-Merging or Concatenating two Dictionaries 
d1 = {'x': 1, 'y': 2}
d2 = {'y': 3, 'z': 4}
d1.update(d2)
print(d1)

#6-Find common elements in three sorted arrays
'''
Input:  ar1 = [1, 5, 10, 20, 40, 80]
            ar2 = [6, 7, 20, 80, 100]
            ar3 = [3, 4, 15, 20, 30, 70, 80, 120]
Output:  [80, 20]
'''
from collections import Counter
def commonElement(ar1,ar2,ar3):
     # first convert lists into dictionary
     ar1 = Counter(ar1)
     ar2 = Counter(ar2)
     ar3 = Counter(ar3)   
     # perform intersection operation
     resultDict = dict(ar1.items() & ar2.items() & ar3.items())
     common = []
     
     # iterate through resultant dictionary  # and collect common elements
     for (key,val) in resultDict.items():
          for i in range(0,val):
               common.append(key)
     print(common)
#7-Find all duplicate characters in string
from collections import Counter
s = "GeeksforGeeks"
# Create a Counter object to count occurrences
# of each character in string
d = Counter(s)  
# Create a list of characters that occur more than once
res = [c for c, cnt in d.items() if cnt > 1]
print(res) 

#8-remove a key from dictionary
a = {"name": "Nikki", "age": 25, "city": "New York"}
# Remove the key 'name' using dictionary comprehension
a = {k: v for k, v in a.items() if k != "name"}
print(a)  

#9-Remove all duplicates words from a given sentence
s1 = "Geeks for Geeks"
s2 = s1.split()  # Split the sentence into words
# Convert the list to a set and back to a list to remove duplicates
s3 = list(set(s2))
# Join the list back into a sentence
s4 = ' '.join(s3)
print(s4)

#10-Counting the Frequencies in a List
a = ['apple', 'banana', 'apple', 'orange', 'banana', 'banana']
# Create an empty dictionary to store the counts
b = {}
# Loop through the list
for c in a:
    # If the item is already in dictionary, increase its count
    if c in b:
        b[c] += 1
    # If the item is not in dictionary, add it with a count of 1
    else:
        b[c] = 1

#11-Possible Words using given characters

def possible_words(Dict, arr):
    arr_set = set(arr)
    result = []
    for word in Dict:
        if set(word).issubset(arr_set):
            result.append(word)
    return result

#12-find second largest number in a list
import heapq

a = [10, 20, 4, 45, 99]

# Get the two largest numbers using heapq.nlargest
top_two = heapq.nlargest(2, a)

# The second largest number is at index 1
print(top_two[1])
 