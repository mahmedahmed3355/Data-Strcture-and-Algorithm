#1-Check if an array is subset of another array
def is_subset(arr1,arr2):
    set2=set(arr2)
    for element in arr1:
        if element not in set2:
            return False
    return True

#2-Check for Disjoint Arrays or Sets
def are_disjoint(set1,set2):
    if not isinstance(set1,set):
        set1=set(set1)
    if not isinstance(set2,set):
        set2=set(set2)
    return set1.isdisjoint(set2)
#3-max distance
def max_distance(arr):
    mp={}
    res=0
    for i in range(len(arr)):
        if arr[i] not in mp:
            mp[arr[i]]=i
        else:
            res=max(res,i-mp[arr[i]])
    return res

#4-union of two array
def union_arrays(arr1, arr2):
    """
    Finds the union of two arrays (or lists).
    Args:
        arr1: The first array.
        arr2: The second array.
    Returns:
        A list containing the union of the two arrays.
    """
    union_set = set(arr1) | set(arr2)  # Use set union operator
    return list(union_set)
#5-Count pairs with absolute difference equal to k
def countpairs(arr,k):
    n=len(arr)
    cnt=0
    for i in range(n):
        for j in range(i+1,n):
            if abs(arr[i]-arr[j])==k:
                cnt+=1
    return cnt
#6-longest subarray with given sum
def longestsubarraywithsum(arr,sum) :
    res = 0
    for i in range(len(arr)) :
        curr = 0 
        for j in range(i,len(arr)) :
            curr += arr[j]
            if (curr == sum) :
                res = max(res,j-i+1)
    
    return res

#7-Finds the length of the longest subarray with an equal number of 0s and 1s.

def longest_subarray_equal_0_1(arr):
    """
    Finds the length of the longest subarray with an equal number of 0s and 1s.
    Args:
        arr: The input array (containing only 0s and 1s).
    Returns:
        The length of the longest subarray with an equal number of 0s and 1s.
    """
    max_length = 0
    prefix_sums = {0: -1}  # Store prefix sums and their indices
    current_sum = 0

    for i, num in enumerate(arr):
        # Treat 0 as -1 and 1 as 1
        current_sum += 1 if num == 1 else -1

        if current_sum in prefix_sums:
            max_length = max(max_length, i - prefix_sums[current_sum])
        else:
            prefix_sums[current_sum] = i

    return max_length
#8-Finds the length of the longest subarray with an equal number of 0s and 1s.
def longest_subarray_equal_0_1(arr):
    """
    Finds the length of the longest subarray with an equal number of 0s and 1s.

    Args:
        arr: The input array (containing only 0s and 1s).

    Returns:
        The length of the longest subarray with an equal number of 0s and 1s.
    """
    max_length = 0
    prefix_sums = {0: -1}  # Store prefix sums and their indices
    current_sum = 0

    for i, num in enumerate(arr):
        # Treat 0 as -1 and 1 as 1
        current_sum += 1 if num == 1 else -1

        if current_sum in prefix_sums:
            max_length = max(max_length, i - prefix_sums[current_sum])
        else:
            prefix_sums[current_sum] = i

    return max_length



#9-Longest Consecutive Subsequence 
def findLongestConseqSubseq(arr, n):

	s = set()
	ans = 0

	# Hash all the array elements
	for ele in arr:
		s.add(ele)

	# check each possible sequence from the start
	# then update optimal length
	for i in range(n):

		# if current element is the starting
		# element of a sequence
		if (arr[i]-1) not in s:

			# Then check for next elements in the
			# sequence
			j = arr[i]
			while(j in s):
				j += 1

			# update optimal length if this length
			# is more
			ans = max(ans, j-arr[i])
	return ans
#10-    Counts the number of distinct elements in all windows of size k in an array.
def count_distinct_in_windows(arr, k):
    """
    Counts the number of distinct elements in all windows of size k in an array.

    Args:
        arr: The input array.
        k: The size of the window.

    Returns:
        A list containing the count of distinct elements for each window.
    """
    if k > len(arr):
        return []

    result = []
    window_counts = {}  # Dictionary to store element counts in the current window

    # Process the first window
    for i in range(k):
        window_counts[arr[i]] = window_counts.get(arr[i], 0) + 1

    result.append(len(window_counts))

    # Process the remaining windows
    for i in range(k, len(arr)):
        # Remove the element that is leaving the window
        window_counts[arr[i - k]] -= 1
        if window_counts[arr[i - k]] == 0:
            del window_counts[arr[i - k]]

        # Add the element that is entering the window
        window_counts[arr[i]] = window_counts.get(arr[i], 0) + 1

        result.append(len(window_counts))

    return result
#11-Checks if there is a subarray with 0 sum in the given array.
def has_zero_sum_subarray(arr):
    """
    Checks if there is a subarray with 0 sum in the given array.
    Args:
        arr: The input array of integers.
    Returns:
        True if there is a subarray with 0 sum, False otherwise.
    """
    prefix_sums = set()  # Store prefix sums encountered
    current_sum = 0
    for num in arr:
        current_sum += num
        if current_sum == 0 or current_sum in prefix_sums:
            return True
        prefix_sums.add(current_sum)
    return False

# Example Usage:
arr1 = [4, 2, -3, 1, 6]
print(has_zero_sum_subarray(arr1))  # Output: True

#12-    Counts the number of subarrays whose elements sum up to the target value.
def count_subarrays_with_sum(arr, tar):
    """
    Counts the number of subarrays whose elements sum up to the target value.

    Args:
        arr: The input array of integers.
        tar: The target sum.

    Returns:
        The number of subarrays with the target sum.
    """
    prefix_sums = {0: 1}  # Store prefix sums and their counts
    current_sum = 0
    count = 0

    for num in arr:
        current_sum += num
        if current_sum - tar in prefix_sums:
            count += prefix_sums[current_sum - tar]

        prefix_sums[current_sum] = prefix_sums.get(current_sum, 0) + 1

    return count
#13-    Checks if two arrays are equal (same elements, same counts).
def are_arrays_equal(a, b):
    """
    Checks if two arrays are equal (same elements, same counts).

    Args:
        a: The first array.
        b: The second array.

    Returns:
        True if the arrays are equal, False otherwise.
    """
    if len(a) != len(b):
        return False

    counts_a = {}
    counts_b = {}

    for num in a:
        counts_a[num] = counts_a.get(num, 0) + 1

    for num in b:
        counts_b[num] = counts_b.get(num, 0) + 1

    return counts_a == counts_b
#14-    Counts the number of subarrays with an equal number of 0s and 1s.

def count_subarrays_equal_0_1(arr):
    """
    Counts the number of subarrays with an equal number of 0s and 1s.

    Args:
        arr: The input array (containing only 0s and 1s).

    Returns:
        The number of subarrays with an equal number of 0s and 1s.
    """
    count = 0
    for i in range(len(arr)):
        for j in range(i, len(arr)):
            subarray = arr[i:j + 1]
            zeros = subarray.count(0)
            ones = subarray.count(1)
            if zeros == ones:
                count += 1
    return count

#15-Check If Array Pair Sums Divisible by k
def can_pairs_be_formed(arr,k):
    if len(arr)%2==1:return False
    count=0
    vis=[-1]*len(arr)
    for i in range(len(arr)):
        for j in range(i+1,len(arr)):
            if (arr[i]+arr[j])%k==0 and vis[i]==-1 and vis[j]==-1:
                count+=1
                vis[i]=1
                vis[j]=1
    return count==len(arr)//2
#16-Longest Subarray With Sum Divisible By K
def longestsubarraydivk(arr,k):
    res=0
    for i in range(len(arr)):
        sum=0
        for j in range(i,len(arr)):
            sum=(sum+arr[j])%k
            if sum==0:
                res=max(res,j-i+1)
    return res
#17-Longest Subarray having Majority Elements Greater Than K
def longestSubarray(arr, k):
    n = len(arr)
    res = 0

    # Traverse through all subarrays
    for i in range(n):
        cnt = 0
        for j in range(i, n):
            if arr[j] > k:
                cnt += 1
            else:
                cnt -= 1

            # Update result with the maximum length
            if cnt > 0:
                res = max(res, j - i + 1)
    
    return res
#18-Longest Subarray with 0 Sum
def max_len(arr):
    
    n = len(arr)
  
    # Initialize the result
    max_len = 0

    # Loop through each starting point
    for i in range(n):
      
        # Initialize the current sum for
        # this starting point
        curr_sum = 0

        # Try all subarrays starting from 'i'
        for j in range(i, n):
          
            # Add the current element to curr_sum
            curr_sum += arr[j]

            # If curr_sum becomes 0, update max_len if required
            if curr_sum == 0:
                max_len = max(max_len, j - i + 1)
    
    return max_len

#19-Subarray with Given Sum – Handles Negative Numbers

def subArraySum(arr, n, sum):

    # Pick a starting point
    for i in range(n):
        curr_sum = 0
        # try all subarrays starting with 'i'
        for j in range(i, n):
            curr_sum += arr[j]
            if (curr_sum == sum):
                print("Sum found between indexes", i, "and", j)
                return

    print("No subarray found")

#20-Partition into minimum subsets of consecutive numbers

def minSubsets(arr):
    
    # Sort the array so that consecutive elements
    # become consecutive in the array.
    arr.sort()

    count = 1
    for i in range(len(arr) - 1):
        
        # Check if there's a break between
        # consecutive numbers
        if arr[i] + 1 != arr[i + 1]:
            count += 1

    return count

if __name__ == "__main__":
    arr = [100, 56, 5, 6, 102, 58, 101, 57, 7, 103, 59]
    print(minSubsets(arr))

#21-Largest subset whose all elements are Fibonacci numbers
def findFibSubset(arr):
    res = []

    # Iterate through all elements of the array
    for num in arr:

        # Using the property of Fibonacci series to check if `num` is a Fibonacci number
        fact1 = 5 * (num ** 2) + 4
        fact2 = 5 * (num ** 2) - 4
        if int(fact1**0.5)**2 == fact1 or int(fact2**0.5)**2 == fact2:
            res.append(num)
        
    return res


# Driver code
if __name__ == "__main__":
    arr = [4, 2, 8, 5, 20, 1, 40, 13, 23]
    res = findFibSubset(arr)

#22-Count Distinct Elements In Every Window of Size K

def countDistinct(arr, k):
    n = len(arr)  
    res = []
  
    # Iterate over every window
    for i in range(n - k + 1):
      
        # Hash Set to count unique elements
        st = set()
        for j in range(i, i + k):
            st.add(arr[j])
      
        # Size of set denotes the number of unique elements
        # in the window
        res.append(len(st))
    return res


if __name__ == "__main__":
    arr = [1, 2, 1, 3, 4, 2, 3]
    k = 4
#23-Group words with same set of characters
from collections import Counter
 
def groupStrings(input):
    # traverse all strings one by one
    # dict is an empty dictionary
    dict={}
     
    for word in input:
        # sort the current string and take it's
        # sorted value as key
        # sorted return list of sorted characters
        # we need to join them to get key as string
        # Counter() method returns dictionary with frequency of
        # each character as value
        wordDict=Counter(word)
 
        # now get list of keys
        key = wordDict.keys()
 
        # now sort these keys
        key = sorted(key)
 
        # join these characters to produce key string
        key = ''.join(key)
         
        # now check if this key already exist in
        # dictionary or not
        # if exist then simply append current word
        # in mapped list on key
        # otherwise first assign empty list to key and
        # then append current word in it
        if key in dict.keys():
            dict[key].append(word)
        else:
            dict[key]=[]
            dict[key].append(word)
 
        # now traverse complete dictionary and print
        # list of mapped strings in each key separated by ,
    for (key,value) in dict.items():
        print (','.join(dict[key]))

#24-Find unique element
'''
Given an array of elements occurring in multiples of k, except one element which doesn't occur in multiple of k. Return the unique element.

Examples:

Input: k = 3, arr[] = [6, 2, 5, 2, 2, 6, 6]
Output: 5
Explanation: Every element appears 3 times except 5.
'''
from collections import Counter
class Solution:
    def find_unique(self, k, arr):
        #code here
        mp = Counter(arr)
        for num, count in mp.items():  # Iterate over items (key-value pairs)
            if count != k:
                return num  # Return the number (key) not the index
        return -1
#25-Count pair sum
'''
Given two sorted arrays arr1 and arr2 of distinct elements. Given a value x. The problem is to count all pairs from both arrays whose sum equals x.
Note: The pair has an element from each array.
Examples:
Input: x = 10, arr1[] = [1, 3, 5, 7], arr2[] = [2, 3, 5, 8] 
Output: 2
Explanation: The pairs are: (5, 5) and (7, 3).  
'''
def countPairs(self,arr1, arr2, x):
    c=0
    k=set(arr2)
    for i in range(len(arr1)):
        p=x-arr1[i]
        if p in k:
            c=c+1
    return c

#26-Difference between highest and lowest occurrence
from collections import Counter
def find_diff(arr):
    counts=Counter(arr)
    if not counts: return 0
    count_value=list(counts.values())
    return max(count_value)-min(count_value)
#26-Substrings with same first and last characters
'''
Given string s, the task is to find the count of all substrings which have the same character 
at the beginning and end.
Example 1:
Input: s = "abcab"
Output: 7
Explanation: a, abca, b, bcab, 
c, a and b
'''
from collections import Counter
def solution(s):
    return sum(f*(f+1)//2 for f in Counter(s).values())

#27-Fake Profile

def solve(self, a):
    vowels = "aeiou"
    consonants = set()
    
    for char in a:
        if char not in vowels:
            consonants.add(char)
                
    if len(consonants) % 2 != 0:
        return "HE!"
    else:
        return "SHE!"
#28-Repeated Character
def firstRep(self, s):
    char_counts = {}  # Use a more descriptive name
    for char in s:
        if char not in char_counts:
            char_counts[char] = 1
        else:
            char_counts[char] += 1
    
    for char in s:
        if char_counts[char] > 1:
            return char
    
    return -1
#29-Uncommon characters
def uncommonChars(self, s1, s2):
    s1_set=set(s1)
    s2_set=set(s2)
    uncommon = sorted((s1_set.symmetric_difference(s2_set)))
    if not uncommon:
        return ""
    else:
        return "".join(uncommon)

#30-Anagram Palindrome
#Check if characters of the given string can be rearranged to form a palindrome.
def isPossible(self, S):
    char_counts = {}
    for char in S:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    odd_counts = 0
    for count in char_counts.values():
        if count % 2 != 0:
            odd_counts += 1
    
    if len(S) % 2 == 0:  # Even length string
        if odd_counts == 0:
            return 1
        else:
            return 0
    else:  # Odd length string
        if odd_counts == 1:
            return 1
        else:
            return 0
#31-Demonitisation
'''
Input:
S = abc
m = ab
n = bc
Output:
-1
Explanation: When we remove the two strings,
we get an empty string and thus the Output -1.
'''
def demonitize(self, S , m , n):
    S=S.replace(m,"")
    S=S.replace(n,"")
    if len(S)>1:
        return S
    return -1
#32-Given a string, say S, print it in reverse manner eliminating the repeated characters and spaces.

class Solution:
    def reverseString(self, s):
        # code here
        seen = set()
        result = ""
        for char in reversed(s):
            if char != ' ' and char not in seen:
                result += char
                seen.add(char)
        return result

#33-Check if a string is Isogram or not
class Solution:
    
    #Function to check if a string is Isogram or not.
    def isIsogram(self,s):
        for i in s:
            if s.count(i)>=2:
                return False
        return True

#34-Count the characters
'''
Given a string S. Count the characters that have ‘N’ number of occurrences. If a character appears consecutively 
it is counted as 1 occurrence.
Example 1:
Input:
S = "abc", N = 1
Output: 3
Explanation: 'a', 'b' and 'c' all have 
1 occurrence.
'''
class Solution:
    def getCount (self,S, N):
        frequency = {}
        prev = None  # To track consecutive characters
        for letter in S:
            if letter != prev:
                frequency[letter] = frequency.get(letter, 0) + 1
            prev = letter
        
        count = sum(1 for key in frequency if frequency[key] == N)
        
        return count

#35-unique numbers
def uniqueNumbers(self, L, R):
    result = []
    for i in range(L, R+1):
        if len(str(i)) == len(set(str(i))):
            result.append(i)
    return result

#36-Given an array of n integers. Find the kth distinct element.
'''
n = 6, k = 2    arr = {1, 2, 1, 3, 4, 2}
Output:  4
Explanation: 1st distinct element will be 3 and the 2nd distinct element will be 4. As 
both the elements present in the array only 1  times.
'''
def KthDistinct(self, arr, k):
    counts = {}
    for num in arr:
        counts[num] = counts.get(num, 0) + 1
    
    distinct_elements = [num for num, count in counts.items() if count == 1]
    
    if k <= len(distinct_elements):
        return distinct_elements[k - 1]
    else:
        return -1

#37-Numbers containing 1, 2 and 3
'''
You are given an array arr[] of integers. Your task is to find all the numbers in the array whose digits consist only of {1, 2, 3}.
Return an array containing only those numbers from arr[]. The order of the numbers in the output array should be the same as 
their order in the input array.
If there is no such element in arr[]. Return {-1}.
Examples:
Input: arr[] = [4, 6, 7]
Output: [-1]
Explanation: No elements are there in the array which contains digits 1, 2 or 3.
'''
class Solution:
    def filterByDigits(self, arr):
        #code here
        lookup={'1','2','3'}
        result=[]
        for element in arr:
            if all(digit in lookup for digit in str(element)):
                result.append(element)
        return result 

#38-Pairs which are Divisible by 4
'''
Given an array arr[] of positive integers, count the number of pairs of integers in the array that have the sum divisible by 4.
Examples:
Input : arr[] = [2, 2, 1, 7, 5]
Output : 3
Explanation: (2,2), (1,7) and (7,5) are the 3 pairs.
'''
class Solution:
    def count4Divisibiles(self, arr ): 
          
        remainders = [0] * 4  # Store counts of remainders (0, 1, 2, 3)
        for num in arr:
            remainders[num % 4] += 1
    
        count = 0
        count += (remainders[0] * (remainders[0] - 1)) // 2  # Pairs with remainder 0
        count += remainders[2] * (remainders[2] - 1) // 2  # Pairs with remainder 2
        count += remainders[1] * remainders[3]  # Pairs with remainders 1 and 3
    
        return count
#39-Non-Repeating Element
class Solution:
    def firstNonRepeating(self, arr): 
        counts={}
        for num in arr:
            counts[num]=counts.get(num,0)+1
        for num in arr:
            if counts[num]==1:
                return num
        return 0

#40-Maximum repeating number
'''
Given an array arr[]. The array contains numbers ranging from 0 to k-1 where k is a positive integer. 
Find the maximum repeating number in this array. If there are two or more maximum repeating numbers, 
return the element with the least value.
Examples:
Input: k = 3, arr[] = [2, 2, 1, 2]
Output: 2
Explanation: 2 is the most frequent element.
'''
from collections import Counter
def maxRepeating(self,k, arr):
    counts = {}
    for num in arr:
        counts[num] = counts.get(num, 0) + 1
    
    max_count = 0
    max_element = float('inf')  # Initialize with positive infinity
    
    for num, count in counts.items():
        if count > max_count:
            max_count = count
            max_element = num
        elif count == max_count and num < max_element:
            max_element = num
    
    return max_element

#41-Completing tasks
'''
Input: arr[] = [2, 5, 6, 7, 9, 4] , k = 15
Output: [[1, 8, 11, 13, 15],[3, 10, 12, 14]] 
Explanation: The remaining tasks are :
{1, 3, 8, 10, 11, 12, 13, 14, 15}.
So Tanya should take these tasks :
{1, 8, 11, 13, 15}.
And Manya should take these tasks :
{3, 10, 12, 14}.
'''
def findTasks(self, arr,k):
    completed_tasks = set(arr)
    remaining_tasks = [task for task in range(1, k + 1) if task not in completed_tasks]
    tanya_tasks = []
    manya_tasks = []
    for i, task in enumerate(remaining_tasks):
        if i % 2 == 0:
            tanya_tasks.append(task)
        else:
            manya_tasks.append(task)
    return [tanya_tasks, manya_tasks]
#42-Last seen array element
'''
Given an array arr[]  of integers that might contain duplicates, 
find the element whose last appearance is earliest.
'''
def lastSeenElement(self, arr): 
    hist = {}
    idx = 0
    for ele in arr:
       # Update the index values
        hist[ele] = idx
        idx += 1
       # Take the min index value and return respective key
    min_val = min(hist.values())
    for k, v in hist.items():
        if v == min_val:
            return k
    return -1
#43-Find all distinct triplets with given sum
def find_distinct_triplets(arr, target_sum):
    """
    Finds all distinct triplets in an array that sum up to the given target sum.
    Args:
        arr: The input array of integers.
        target_sum: The target sum to find.
    Returns:
        A list of distinct triplets (tuples) that sum up to the target sum.
    """
    arr.sort()  # Sort the array to efficiently find triplets
    triplets = set()  # Use a set to store distinct triplets
    for i in range(len(arr) - 2):
        left = i + 1
        right = len(arr) - 1
        while left < right:
            current_sum = arr[i] + arr[left] + arr[right]
            if current_sum == target_sum:
                triplets.add(tuple(sorted((arr[i], arr[left], arr[right])))) #add tuple to set
                left += 1
                right -= 1
            elif current_sum < target_sum:
                left += 1
            else:
                right -= 1

    return list(triplets)
#44-Numbers with prime frequencies greater than or equal to k
'''
Input: arr[] = [11, 11, 11, 23, 11, 37, 51, 37, 37, 51, 51, 51, 51], k = 2
Output: [37, 51]
Explanation: 11's count is 4, 23 count 1, 37 count 3, 51 count 5. 37 and 51 are 
two number that appear prime number of time and frequencies greater than or 
equal to k = 2.
'''
from collections import Counter
class Solution:
    # Function to find numbers with prime occurrences
    def primeOccurences(self, arr, k):
        def isPrime(x):
            if x < 2:
                return False
            for i in range(2, int(x**0.5)+1):
                if x % i == 0:
                    return False
            return True
            
        hist = {}
        for ele in arr:
            hist[ele] = hist.get(ele,0)+1.
        res = []
        for key, val in hist.items():
            if val >= k:
                if isPrime(val):
                    res.append(key)
        return sorted(res)
#45-Cumulative frequency of count of each element in an unsorted array
from collections import Counter
class Solution:
    def countFreq(self, arr):
        cnt=Counter(arr)
        #Complete the function
        sorted_item=sorted(cnt.items())
        res=0
        cmm=[]
        for _,freq in sorted_item:
            res+=freq
            cmm.append(res)
        return cmm
#46-Max distance between same elements
def maxDistance(self, arr):
	freq_dict={}
        max_dist=0
        for i,num in enumerate(arr):
            if num in freq_dict:
                max_dist=max(max_dist,i-freq_dict[num])
            else:
                freq_dict[num]=i
        return max_dist
#47-Largest Fibonacci Subsequence
    def findFibSubset(self, arr):
        fib_set = set()
        maxN = max(arr)
        a = 0
        b = 1
        while a <= maxN:
            fib_set.add(a)
            a,b = b, a+b
        result = [num for num in arr if num in fib_set]
        return result
#48-find all triplest for given sum
    def threeSum(self, arr, target):
        # Your code here
        
        arr.sort()
        res=[]
        for i in range(len(arr)):
            for j in range(i+1,len(arr)):
                for k in range(j+1,len(arr)):
                    if arr[i]+arr[j]+arr[k]==target:
                        res.append([arr[i],arr[j],arr[k]])
        return res
        
#49-Subarrays with given sum
'''
Input: arr[] = [10, 2, -2, -20, 10], k = -10
Output: 3
Explanation: Subarrays with sum -10 are: [10, 2, -2, -20], [2, -2, -20, 10] and 
[-20, 10].
'''
class Solution:
    def subArraySum(self,arr,k):  
        count = 0
        curr_sum = 0
        hashmap = {}  # To store the cumulative sum and its frequency
    
        # Initialize hashmap with 0 sum and 1 frequency
        hashmap[0] = 1
    
        for num in arr:
            curr_sum += num
            if curr_sum - k in hashmap:
                count += hashmap[curr_sum - k]
            hashmap[curr_sum] = hashmap.get(curr_sum, 0) + 1
    
        return count
#50-Find pairs with given relation
'''
Input: arr[] = [3, 4, 7, 1, 2, 9, 8]
Output: 1
Explanation: Product of 4 and 2 is 8 and also,the product of 1 and 8 is 8.
'''
def find_pairs(arr):
	product_map={}
	for i in range(len(arr)):
		for j in range(i+1,len(arr)):
			product=arr[i]*arr[j]
			if product in product_map:
				prev_pair=product_map[product]
				if prev_pair[0] != arr[i] and prev_pair[0] != arr[j] and prev_pair[1] != arr[i] and prev_pair[1] != arr[j]:
                        		return 1
                	else:
                    		product_map[product] = (arr[i], arr[j])
    
        return -1

#51-Count pairs with given sum
'''
Input: arr[] = [1, 5, 7, -1, 5], target = 6 
Output: 3
Explanation: Pairs with sum 6 are (1, 5), (7, -1) and (1, 5). 
'''
    #Complete the below function
    def countPairs(self,arr, target):
        #Your code here
        count = 0
        hashmap = {}  # To store the elements and their frequencies
    
        for num in arr:
            complement = target - num
            if complement in hashmap:
                count += hashmap[complement]
            hashmap[num] = hashmap.get(num, 0) + 1
    
        return count
#52-Count pairs Sum in matrices
'''
Input: 
n = 3, x = 21
mat1 = {{1, 5, 6},
        {8, 10, 11},
        {15, 16, 18}}
mat2 = {{2, 4, 7},
        {9, 10, 12},
        {13, 16, 20}}
OUTPUT: 4
Explanation: The pairs whose sum is found to be 21 are (1, 20), (5, 16), (8, 13), (11, 10).
'''
class Solution:
	def countPairs(self, mat1, mat2, n, x):
        # code here
        c=0
        l1,l2=[],[]
        for i in range(len(mat2)):
            for j in range(len(mat2[0])):
                l2.append(mat2[i][j])
                l1.append(mat1[i][j])
        l2=set(l2)
        for i in l1:
            if x-i in l2:
                c+=1
        return c    
#53-Minimum Distinct Ids
'''
Given an array of items, the i'th index element denotes the item id’s and given a number m, the task is to remove m elements such that there should be minimum distinct id’s left. Print the number of distinct id’s.

Example 1 -

Input:
n = 6
arr[] = {2, 2, 1, 3, 3, 3}
m = 3
Output:
1
Explanation : 
Remove 2,2,1
'''
class Solution:
    def distinctIds(self,arr : list, n : int, m : int):
        # Complete this function
        counts=Counter(arr)
        frequiences=sorted(counts.values())
        removed=0
        distinict_id=len(counts)
        for freq in frequiences:
            if removed+freq<=m:
                removed+=freq
                distinict_id-=1
            else:
                break
        return distinict_id
#54-Powerfull Integer
'''
You are given a 2D integer array of intervals whose length is n where intervals[i]=[start, end] I.e. all integers from start to end inclusive of start and end are also present and also we are given an integer k. We have to return the Powerfull Integer. A powerful Integer is an integer that occurs at least k times. If multiple integers have at least k occurrences, we have to return the maximum integer out of all those elements. 

Note: If no integer occurs at least k times return -1.

Example 1:

Input :
n=3
intervals={{1,3},{4,6},{3,4}}
k=2
Output: 4
Explanation:
As we can see that 3 and 4 are the 2 integers 
that have 2 occurences(2>=k) so we have 4 
in this case as the Powerfull integer as it 
is the maximum element which satisfies the condition.
'''
class Solution:
    def powerfullInteger(self, n : int, intervals : List[List[int]], k : int) -> int:
        d = {}
        for s,e in intervals :
            d[s] = d.get(s,0)+1
            d[e] = d.get(e,0)
            d[e+1] = d.get(e+1,0) - 1
    
        ans = -1
        cur = 0
        for i in sorted(d) :
            cur += d[i]
            if cur >= k :
                ans= i
        
        return (ans)
#55-Split array into minimum subsets
'''
Given an array arr[] of distinct positive numbers. Split the array into the minimum number of subsets (or subsequences) such that each subset contains consecutive numbers.

Examples:

Input: arr = [100, 56, 5, 6, 102, 58, 101, 57, 7, 103, 59]
Output: 3
Explanation: [5, 6, 7], [56, 57, 58, 59], [100, 101, 102, 103] are 3 subsequences in which numbers are consecutive.
'''
class Solution:
    #Complete the below function
    def minSubsets(self, arr):
        #Your code here
        
        uarr = set(arr)
        res = 0
        for a in arr:
            if a - 1 not in uarr:
                res += 1
        return res
#56-K-Pangrams
'''
Given a string str and an integer k, return true if the string can be changed into a pangram after at most k operations, else return false.

A single operation consists of swapping an existing alphabetic character with any other lowercase alphabetic character.

Note - A pangram is a sentence containing every letter in the english alphabet.

Examples :

Input: str = "the quick brown fox jumps over the lazy dog", k = 0
Output: true
Explanation: the sentence contains all 26 characters and is already a pangram.
Input: str = "aaaaaaaaaaaaaaaaaaaaaaaaaa", k = 25 
Output: true
Explanation: The word contains 26 instances of 'a'. Since only 25 operations are allowed. We can keep 1 instance and change all others to make str a pangram.
'''
    def kPangram(self,s, k):
    # code here
        unique_chars = set(c for c in s if c.isalpha())

        # Count how many letters are missing to form a pangram
        missing_chars = 26 - len(unique_chars)

        # If missing characters are more than k OR if we don't have enough letters, return False
        return missing_chars <= k and len(s.replace(" ", "")) >= 26
#57-Minimum indexed character
'''
Given a string s1 and another string s2. Find the minimum index of the character in s1 that is also present in s2.
Examples :
Input: s1 = "geeksforgeeks", s2 = "set"
Output: 1
Explanation: e is the character which is present in given s1 "geeksforgeeks" and is also present in s2 "set". Minimum index of e is 1. 
'''
class Solution:
    
    #Function to find the minimum indexed character.
    def minIndexChar(self,s1,s2):
        char_set=set(s2)
        for i,char in enumerate(s1):
            if char in char_set:
                return i
                
        return -1

#58-Positive Negative Pair
'''
Given an array of distinct integers, find all the pairs having both negative and positive values of a number in the array.


Example 1:

Input:
n = 8
arr[] = {1,3,6,-2,-1,-3,2,7}
Output: -1 1 -3 3 -2 2
Explanation: 1, 3 and 2 are present 
pairwise positive and negative. 6 and 
7 have no pair.
'''
class Solution:
    #Function to return list containing all the pairs having both
    #negative and positive values of a number in the array.
    def findPairs(self,arr,n):
        d = {}
        l = []
        for i in arr:
            if i in d:
                l.append(-abs(i))
                l.append(abs(i))
            else:
                d[i] = 1
                d[-i] = 1
        return l
#59-Maximum number of characters between any two same character
'''
Given a string containing lower and uppercase alphabets, the task is to find the maximum number of characters between any two same (case sensitivity) character in the string.

Example 1:

Input:
S = "socks"
Output: 3
Explanation: There are 3 characters between
the two occurrences of 's'.
'''
from collections import Counter
class Solution:

    def maxChars(self, S):
        char_indices = {}
        max_distance = -1

        for i, char in enumerate(S):
            if char in char_indices:
                distance = i - char_indices[char] - 1
                max_distance = max(max_distance, distance)
            else:
                char_indices[char] = i
    
        return max_distance
#60-Remove minimum number of elements
'''
Difficulty: EasyAccuracy: 24.34%Submissions: 5K+Points: 2
Given two arrays arr1[] and arr2[], the task is to find the minimum number of elements to remove from each array such that no common elements exist between the two arrays.

Examples:

Input: arr1[] = [2, 3, 4, 5, 8], arr2[] = [1, 2, 3, 4]
Output: 3
Explanation: To remove all common elements, we need to delete 2, 3, and 4 from either array.
'''
from collections import Counter
class Solution:
    def minRemove(self, arr1, arr2):
        count1 = Counter(arr1)
        count2 = Counter(arr2)
        
        # Find common elements
        common_elements = set(arr1) & set(arr2)
        
        # Initialize the removal count to 0
        removals = 0
        
        # For each common element, add the minimum count of occurrences in both arrays
        for element in common_elements:
            removals += min(count1[element], count2[element])
        
        return removals
#61-Find All Pairs With Given Sum
'''
Given a 0 indexed array arr[] and a target value, the task is to find all possible indices (i, j) of pairs (arr[i], arr[j]) whose sum is equal to target and i != j.

Note: Return the list of pairs sorted lexicographically by the first element, and then by the second element if necessary.

Examples:

Input: arr[] = [10, 20, 30, 20, 10, 30], target = 50 
Output: [[1, 2], [1, 5], [2, 3], [3, 5]]
Explanation: All pairs with sum = 50 are:
arr[1] + arr[2] = 20 + 30 = 50 
arr[1] + arr[5] = 20 + 30 = 50 
arr[2] + arr[3] = 30 + 20 = 50 
arr[3] + arr[5] = 20 + 30 = 50
'''
class Solution:
    def findAllPairs(self, arr, target):
        n = len(arr)
        res = []
        hist = {}
        for i in range(n):
            if (target - arr[i]) in hist:
                for index in hist[target - arr[i]]:
                    res.append([index, i])
            if arr[i] not in hist:
                hist[arr[i]] = []
            hist[arr[i]].append(i)
        res.sort()
        return res

#62-Count the elements
'''
Given two arrays a and b both of size n. Given q queries in an array query each having a positive integer x denoting an index of the array a. For each query, your task is to find all the elements less than or equal to a[x] in the array b.

Example 1:

Input:
n = 3
a[] = {4,1,2}
b[] = {1,7,3}
q = 2
query[] = {0,1}
Output : 
2
1
Explanation: 
For 1st query, the given index is 0, a[0] = 4. There are 2 elements(1 and 3) which are less than or equal to 4.
For 2nd query, the given index is 1, a[1] = 1. There exists only 1 element(1) which is less than or equal to 1.
'''
from bisect import bisect_right
class Solution:
    def countElements(self, a, b, n, query, q):
        # code here
        res = []
        b.sort()
        
        for i in query:
            count = bisect_right(b,a[i])
            res.append(count)
            
        return res

#63-2 Sum – Count distinct pairs with given sum
'''
Given an array arr[] and an integer target. You have to find numbers of distinct pairs in array arr[] which sums up to given target. 

Note: (a, b) and (b, a) are considered the same. Also, the same numbers at different indices are considered same.

Examples:

Input: arr[] = [5, 6, 5, 7, 7, 8], target = 13 
Output: 2
Explanation: Distinct pairs with sum equal to 13 are (5, 8) and (6, 7).
Input: arr[] = [2, 6, 7, 1, 8, 3], target = 10 
Output: 2
Explanation: Distinct pairs with sum equal to 10 are (2, 8) and (7, 3).
'''
class Solution:
    #Complete the below function
    def countDistinctPairs(self,arr, target):
        seen = set()
        pairs = set()
    
        for num in arr:
            complement = target - num
            if complement in seen:
                pair = tuple(sorted((num, complement))) #ensure (a,b) and (b,a) are same
                pairs.add(pair)
            seen.add(num)
    
        return len(pairs)

#64-Sum Indexes
'''
You are given 3 different arrays A, B, and C of the same size N. Find the number of indexes i such that:
Ai = Bi + Ck 
where k lies between [1, N].

 

Example 1:

Input: N = 3
A = {1, 2, 3}
B = {3, 2, 4}
C = {0, 5, -2}
Output: 2
Explaination: The possible i's are 0 and 1. 
1 = 3 + (-2) and 2 = 2 + 0.
'''
class Solution:
    def pairCount(self, N, A, B, C):
        # code here
        count=0
        for i in range(N):
            if A[i]-B[i] in C:
                count+=1
        return count

#65-Convert a sentence into its equivalent mobile numeric keypad sequence

'''
Input:
S = "GFG"
Output: 43334
Explanation: For 'G' press '4' one time.
For 'F' press '3' three times.
'''
class Solution:

    def printSequence(self,sentence):
        # code here
        keypad = {
            'A': '2', 'B': '22', 'C': '222',
            'D': '3', 'E': '33', 'F': '333',
            'G': '4', 'H': '44', 'I': '444',
            'J': '5', 'K': '55', 'L': '555',
            'M': '6', 'N': '66', 'O': '666',
            'P': '7', 'Q': '77', 'R': '777', 'S': '7777',
            'T': '8', 'U': '88', 'V': '888',
            'W': '9', 'X': '99', 'Y': '999', 'Z': '9999',
            ' ': '0'
        }
    
        result = ""
        for char in sentence.upper():
            if char in keypad:
                result += keypad[char]
            elif char.isalpha():
                # Handle characters not in the keypad (e.g., accented characters)
                pass
            else:
                if char == ' ':
                    result += '0'
                elif char.isalnum():
                    pass
                else:
                    pass
    
        return result
#66-Count Non-Repeated Elements
'''
You are given an array of integers arr[]. You need to print the count of non-repeated elements in the array.

Example 1:

Input: arr[] = [1, 1, 2, 2, 3, 3, 4, 5, 6, 7]
Output: 4
Explanation: 4, 5, 6 and 7 are the elements with frequency 1 and rest elements are repeated so the number of non-repeated elements are 4.
Input: arr[] = [10, 20, 30, 40, 10]
Output: 3
Explanation: 20, 30, 40 are the elements with the frequency 1 and 10 is the repeated element to number of non-repeated elements are 3.
'''
class Solution:
    
    #Complete this code
    #Function to return the count of non-repeated elements in the array.
    def countNonRepeated(self,arr):
        #Your code here
        counts = {}
        for num in arr:
            counts[num] = counts.get(num, 0) + 1
    
        non_repeated_count = 0
        for count in counts.values():
            if count == 1:
                non_repeated_count += 1
    
        return non_repeated_count

#67-Distinct Substrings
'''
Given a string s consisting of uppercase and lowercase alphabetic characters. Return the  number of distinct substrings of size 2 that appear in s as contiguous substrings.

Example

Input :
s = "ABCAB"
Output :
3
Explanation:  For "ABCAB", the 
three distinct substrings of size 
2 are "AB", "BC" and "CA". 
'''
class Solution:

    def fun(self, s):
        # code here
        if len(s) < 2:
            return 0
    
        substrings = set()
        for i in range(len(s) - 1):
            substrings.add(s[i:i + 2])
    
        return len(substrings)
#68-Grouping values
'''
Grouping values
Difficulty: EasyAccuracy: 40.92%Submissions: 3K+Points: 2
There are N integers given in an array arr. You have to determine if it is possible to divide them in at most K groups, such that any group does not have a single integer more than twice.

Example 1:

Input: N = 5, K = 2
arr = {1, 1, 2, 5, 1}
Output: 1
Explaination: We can divide them in two groups in 
following way (1, 1, 2, 5) and (1).
Example 2:

Input: N = 3, K = 1
arr = {1, 1, 1}
Output: 0
Explaination: There can be only one group and 
three 1's. So it is not possible.
'''
from collections import Counter
class Solution:
    def isPossible(self, N, arr, K):
        # code here
        freq=Counter(arr)
        max_count = 0
        for count in freq.values():
            max_count = max(max_count, count)

        if max_count > 2 * K:
            return 0
        else:
            return 1
#69-Luckely number
def is luckely(num):
	num_str=str(num)
	digit_products=set()
	for i in range(len(num_str):
		for j in range(i+1,len(num_str)+1):
			sub_num=num[i:j]
			product=1
			for digit in sub_num:
				product*=int(digit)
			digit_products.add(product)
	if len(digit_products)==len(num_str)*len(num_str)+1)//2:
		return 1
	else:
		return 0

#70-smallest subarray with all occ of amost frequent element
'''
Input : arr[] = [1, 2, 2, 3, 1]
Output : [2, 2]
Explanation: Note that there are two elements that appear two times, 1 and 2. The smallest window for 1 is whole array and smallest window for 2 is [2, 2]. Since window for 2 is smaller, this is our output.
'''
from collections import Counter
def smallestsubsegment(arr):
	counts=Counter(arr)
	max_count=max(counts.values())
	winners=set(x,k in counts.items() if k=max_count)
	first,last={},{}
	for i,x in enumerate(arr):
		if x in winners:
			if x not in first:
				first[x]=i
				last[x]=i
	max_start=0
	max_stop=max_size=len(arr)
	for x,start in first.items():
		stop=last[x]+1
		if stop-start<max_size:
			max_start,max_stop,max_size=start,stop,stop-start
	return arr[max_start:max_stop]
#71-Top K Frequent in Array
'''
Input: arr[] = [3, 1, 4, 4, 5, 2, 6, 1], k = 2
Output: [4, 1]
Explanation: Frequency of 4 is 2 and frequency of 1 is 2, these two have the maximum frequency and 4 is larger than 1.
'''
from collections import Counter
class Solution:
    def topKFrequent(self, arr, k):
        # Step 1: Count frequencies
        freq_map = Counter(arr)  # O(N)
        
        # Step 2: Sort by frequency (descending), then by value (descending)
        sorted_items = sorted(freq_map.keys(), key=lambda x: (-freq_map[x], -x))  # O(N log N)

        # Step 3: Return the top K elements
        return sorted_items[:k]  # O(K)
#72-Absolute difference divisible by K
'''
Input:
n = 3
arr[] = {3, 7, 11}
k = 4
Output:
3
Explanation:
(11-3) = 8 is divisible by 4
(11-7) = 4 is divisible by 4
(7-3) = 4 is divisible by 4
'''
class Solution:
    def countPairs (self, n, arr, k):
        # code here
        list =[]
        for i in range(k):
            list.append(0)
        for i in arr:
            r = i%k
            list[r]+=1
        maxx=0
        for i  in list:
            maxx+= (i*(i-1))//2
        return maxx
#73-Overlapping Intervals
'''
Input: arr[][] = [[1,3],[2,4],[6,8],[9,10]]
Output: [[1,4], [6,8], [9,10]]
Explanation: In the given intervals we have only two overlapping intervals here, [1,3] and [2,4] which on merging will become [1,4]. Therefore we will return [[1,4], [6,8], [9,10]].
'''
	def mergeOverlap(self, intervals):
		#Code here
    	if not intervals:
            return []
    
        intervals.sort(key=lambda x: x[0])  # Sort intervals by start time
        merged = [intervals[0]]
    
        for start, end in intervals[1:]:
            last_end = merged[-1][1]
            if start <= last_end:
                merged[-1][1] = max(end, last_end)
            else:
                merged.append([start, end])
    
        return merged
#73-Maximum possible sum
'''
Given two arrays arr1 and arr2, the task is to find the maximum sum possible of a window in array arr2 such that elements of the same window in array arr1 are unique.

Examples:

Input: arr1 = [0, 1, 2, 3, 0, 1, 4], arr2 = [9, 8, 1, 2, 3, 4, 5] 
Output: 20
Explanation: The maximum sum occurs for the window [9, 8, 1, 2] in arr2, which corresponds to the window [0, 1, 2, 3] in arr1 where all elements are unique. The sum is 9 + 8 + 1 + 2 = 20.
'''
    def returnMaxSum(self, arr1, arr2):
        # cdoe here
        n = len(arr1)
        left = 0
        max_sum = 0
        current_sum = 0
        element_index = {}

        for right in range(n):
            if arr1[right] in element_index and element_index[arr1[right]] >= left:
                left = element_index[arr1[right]] + 1
                current_sum = sum(arr2[left:right + 1])
            else:
                current_sum += arr2[right]
            element_index[arr1[right]] = right
            max_sum = max(max_sum, current_sum)
        return max_sum
#75-Count pairs in array divisible by K
'''
Given an array arr[] and positive integer k, the task is to count total number of pairs in the array whose sum is divisible by k.

Examples:

Input :  arr[] = {2, 2, 1, 7, 5, 3}, k = 4
Output : 5
Explanation : There are five pairs possible whose sum is divisible by '4' i.e., (2, 2), (1, 7), (7, 5), (1, 3) and (5, 3)
'''
from collections import defaultdict
class Solution:
    def countKdivPairs(self, arr, n, k):
        #code here
        count = 0
        dic = defaultdict(int)
        for i in arr:
            r = i % k
            if r == 0:
                count += dic[0]
            else:
                count += dic[k-r]
            dic[r] += 1
        return count
#78-Sort Elements by Decreasing Frequency
'''
Input: arr[] = [5, 5, 4, 6, 4]
Output: [4, 4, 5, 5, 6]
Explanation: The highest frequency here is 2. Both 5 and 4 have that frequency. Now since the frequencies are the same the smaller element comes first. So 4 4 comes first then comes 5 5. Finally comes 6. The output is 4 4 5 5 6.
'''
    def sortByFreq(self,arr):
        d = {}
        res = []
         for i in arr:
            if i in d.keys():
                d[i] += 1
            else:
                d[i] = 1
        sorted_items = sorted(d.items(), key=lambda x: (-x[1], x[0]))
        # Creating the result list
        for key, value in sorted_items:
            res.extend([key] * value)  # Append key 'value' times

        return res
#79-Subarray range with given sum
'''
Input: arr[] = [10, 2, -2, -20, 10] , tar = -10
Output: 3
Explanation: Subarrays with sum -10 are: [10, 2, -2, -20], [2, -2, -20, 10] and [-20, 10].
'''
    def subArraySum(self,arr, tar):
        #Your code here
        s={0:1}
        cur,res=0,0
        for i in arr:
            cur+=i
            if cur-tar in s:
                res+=s[cur-tar]
            s[cur]=s.get(cur,0)+1
        return res

#80-k-Anagram
'''
Two strings are called k-anagrams if both of the below conditions are true.
1. Both have same number of characters.
2. Two strings can become anagram by changing at most k characters in a string.

Given two strings of lowercase alphabets and an integer value k, the task is to find if two strings are k-anagrams of each other or not.

Example:

Input: s1 = "fodr", s2 = "gork", k = 2
Output: true
Explanation: We can change 'f' -> 'g' and 'd' -> 'k' in s1.
'''
from collections import Counter

class Solution:
    def areKAnagrams(self, s1, s2, k):
        if len(s1) != len(s2):
            return False  # Strings must be of the same length

        # Step 1: Compute frequency count of both strings
        freq_s1 = Counter(s1)
        freq_s2 = Counter(s2)

        # Step 2: Count excess characters in s1 that need to be changed
        required_changes = 0

        for char in freq_s1:
            if freq_s1[char] > freq_s2[char]:  # Extra characters in s1
                required_changes += freq_s1[char] - freq_s2[char]

        # Step 3: If required changes are within k, return True
        return required_changes <= k
#81-Print Anagrams Together
'''
Given an array of strings, return all groups of strings that are anagrams. The strings in each group must be arranged in the order of their appearance in the original array. Refer to the sample case for clarification.

Examples:

Input: arr[] = ["act", "god", "cat", "dog", "tac"]
Output: [["act", "cat", "tac"], ["god", "dog"]]
Explanation: There are 2 groups of anagrams "god", "dog" make group 1. "act", "cat", "tac" make group 2.
'''
class Solution:

    def anagrams(self, arr):
        group_map = {}
        for i in arr:
            w = "".join(sorted(i))
            if w in group_map:
                group_map[w].append(i)
            else:
                group_map[w] = [i]
        return group_map.values()
