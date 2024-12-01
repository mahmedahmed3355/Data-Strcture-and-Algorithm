#two sum where input array is sorted
def twosum(numbers,target):
    l,r=0,len(numbers)-1
    while l<r:
        if numbers[l]+numbers[r]==target:
            return[l+1,r+1]
        elif numbers[l]+numbers[r]<target:
            l+=1
        else:
            r-=1
    return []

#2- 3 sum
def threesum(nums):
    h,n,s={},len(nums),set()
    for i,um in enumerate(nums):
        h[nums]=i
    for i in range(nums):
        for j in range(i+1,n):
            desired=-nums[i]-nums[j]
            if desired in h and h[desired]!=i and h[desired]!=j:
                s.add(tuple(sorted(nums[i],nums[j],desired)))
    return s

#3-4 sum
def four_sum(nums,target):
    if not nums or len(nums)<4: return []
    nums.sort()
    res,n=set(),len(nums)
    for i in range(n):
        for j in range(i+1,n):
            l,r=j+1,n-1
            if target==nums[i]+nums[j]+nums[r]+nums[l]:
                res.add(nums[i],nums[j],nums[r],nums[l])
                l+=1
                r-=1
            elif target>nums[i]+nums[j]+nums[r]+nums[l]:
                l+=1
            else:
                r-=1
    return res

#4-number of subsequence that satisfy the given sum condition
def numofsubsequence(nums,target):
    ans=0
    nums.sort()
    i,j=0,len(nums)-1
    while(i<=j):
        if nums[i]+nums[j]<=target:
            ans+=pow(2,(j-i),1000000007)
            i+=1
        else:
            j-=1
    return ans%1000000007

#5-sum of square numbers
from math import isqrt,sqrt
def judgesquaresum(c):
    for a in range(isqrt(c)+1):
        b2=c-a*a
        if sqrt(b2)%1==0:
            return True
    return False

#6- boat to save people
def numrescueboats(people,limt):
    people.sort()
    left,right=0,len(people)-1
    boats=0
    while left<=right:
        if people[left]+people[right]<=limt:
            left+=1
        right-=1
        boats+=1
    return boats

#7-minimize maximum pair sum in array
def minpairsum(nums):
    nums.sort()
    mid=len(nums)//2
    return max(a+b for a,b in zip(nums[:mid],nums[:mid-1:-1]))      

#8- 3sum with multiplicty
from collections import Counter
def threesummulti(arr,target):
    n,toreturn=len(arr),0
    nums=Counter(arr[:1])
    for j in range(1,n-1):
        for k in range(j+1,n):
            toreturn=(toreturn+nums[target-arr[k]-arr[j]])%1000000007
        nums[arr[j]]+=1
    return toreturn

#9-Next permutation
def nextpermutation(nums):
    i=len(nums)-2
    while i>=0 and nums[i]>=nums[i+1]:
        i-=1
    if i>=0:
        j=len(nums)-1
        while nums[j]<=nums[i]:
            j-=1
        nums[i],nums[j]=nums[j],nums[i]
    nums[i+1:]=reversed(nums[i+1:])

#10-next greater element
def nextgreaterelement(n):
    cs=list(str(n))
    n,i,j=len(cs),n-2,n-1
    while i>=0 and cs[i]>=cs[i+1]:
        i-=1
    if i<0:
        return -1
    while cs[i]>=cs[j]:
        j-=1
    cs[i],cs[j]=cs[j],cs[i]
    cs[i+1:]=cs[i+1:][::-1]
    ans=int("".join(cs))
    return -1 if ans>2**31-1 else ans

#11- longest substring without repeating charcter
def longestsubstring(s):
    seen,l,max_length =set(),0,0
    for r,char in enumerate(s):
        while char in seen:
            seen.remove(s[l])
            l+=1
        seen.add(char)
        max_length=max(max_length,r-l+1)
    return max_length

#12-longest substring with at most two distinct charcter
def longestsubstringwithdistinictcharcter(s):
    seen,ans,max_len=set(),"",-1
    for i in range(len(s)):
        for j in range(i+1,len(s)+1):
            seen=set(s[i:j])
            if len(seen)==2:
                max_len=max(max_len,j-i)
    return max_len

#13-longest consequative subcharcter
def solution(s):
    maxx,count=1,1
    for i in range(len(s)):
        if s[i]==s[i+1]:
            count+=1
            maxx=max(maxx,count)
        else:
            count=1
    return maxx

#14-longest increasing subsequence
def lis(nums):
    if not nums : return 0
    res,curr=0,0
    for i in range(len(nums)):
        if nums[i]>nums[i-1]:
            curr+=1
            res=max(res,curr)
        else:
            curr=1
    return res

#15-reverse string with special charcter
def reversewitjspecialcharcter(s):
    s,r,l=list(s),0,len(s)-1
    while l<r:
        if not s[l].isalpha():
            l+=1
        if not s[r].isalpha():
            r-=1
        else:
            s[l],s[r]=s[r],s[l]
            l+=1
            r-=1
    return "".join(s)
    
#16-reverse words in string
def reversewordinstring(s):
    return "".join(reversed(s.split()))

#17-reverse word in string 2
def refersewordsinstring2(s):
    def reverse(i,j):
        while i<j:
            s[i],s[j]=s[j],s[i]
            i,j=i+1,j-1
    i,n=0,len(s)
    for j,c in enumerate(s):
        if c=="":
            reverse(i,j-1)
            i=i+1
        elif j==n-1:
            reverse(i,j)
    reverse(0,n-1)

#18-reverse words in string3
def reversewords(s):
    return "".join(word[::-1] for word in s.split(""))

#19-reverse string
def reversestring(s):
    l,r=0,len(s)-1
    while l<r:
        s[l],s[r]=s[r],s[l]
        l,r=l+1,r-1

#20-reverse string 2
def reversestr(s,k):
    lst=list(s)
    for i in range(0,len(lst),2*k):
        lst[i:i+k]=reversed(lst[i:i+k])
    return "".join(lst)

#21-reverse vowels in string
def revesevowels(s):
    vowels="AEIOUaeiou"
    vews=[c for c in s if c in vowels]
    revstr=[]
    for char in s:
        if char in vowels:
            revstr.append(vews.pop())
        else:
            revstr.append(char)
    return "".join(revstr)

#22-reverse only letters
def reverse_only_letters(s):
    cs=list(s)
    i,j=0,len(cs)-1
    while i<j:
        while i<j and not cs[i].isalpha():
            i+=1
        while i<j and not cs[i].isalpha():
            j-=1
        if i<j:
            cs[i],cs[j]=cs[j],cs[i]
            i,j=i+1,j-1
    return "".join(cs)

#23-reverse string between each pair of parenthes
def reversepathrness(s):
    stk=[]
    for c in s:
        if c==")":
            t=[]
            while stk[-1]!="(":
                t.append(stk.pop())
            stk.pop()
            stk.append(t)
        else:
            stk.append(c)
    return "".join(stk)

#24-reverse prefix in string
def reverseprefixword(word,ch):
    i=word.find(ch)
    if i !=-1:
        return word[:i+1][::-1]+word[i+1:]
    return word

#25-revere words with dot
def reverse_words(str):
    input_word=str.split(".")
    reversed_word=input_word[::-1]
    return ".".join(reversed_word)

#26-reverse string
def reverseinteger(x):
    sign=-1 if x<0 else 1
    x*=sign
    reversed_x=int(str(x)[::-1])
    reversed_x*=sign
    if reversed_x<-2**31 or reversed_x>2**31-1:
        return 0
    return reversed_x

#27-vaild palindrome
def ispalindrome(s):
    s=s.lower()
    s="".join(filter(str.isalnum(s)))
    return 1 if s==s[::-1] else 0

#28-vaild palindrome 2
def vaildpalindrome(s):
    if s==s[::-1] : return True
    l,r=0,len(s)-1
    while l<=r:
        if s[l]!=s[r]:
            temp2=s[:r]+s[r+1:]
            temp=s[:l]+s[l+1:]
            return temp2==temp[::-1] or temp[::-1]==temp2[::-1]
        l+=1
        r-=1

#29-vaild palindrome3
# mak string palindrome by removinr one char return T/F
def makepalindrome(s):
    i,j=0,len(s)-1
    cnt=0
    while i<j:
        cnt+=s[i]!=s[j]
        i,j=i+1,j-1
    return cnt<=2
#30-longest palindrome substring
def longest_palindrome(s):
    longest,i,j="",0,0
    for i in range(len(s)):
        for j in range(i+1,len(s)+1):
            if s[i:j]==s[i:j][::-1]:
                if len(s[i:j])>len(longest):
                    longest=s[i:j]
    return longest

#31-count palindrome substring
def countpalindrome(s):
    counter,i,j=0,0,0
    for i in range(len(s)):
        for j in range(i+1,len(s)+1):
            if s[i:j]==s[i:j][::-1]:
                counter+=1
    return counter

#32-convert to palindrome
def slove(a):
    l,r=0,len(a)-1
    while l<r:
        if a[l]==a[r]:
            l+=1
            r-=1
        else:
            if a[l+1:r+1]==a[l+1:r+1][::-1]:
                return 1
            if a[l:r]==a[l:r][::-1]:
                return 1
            return 0
    return 1

#33-shortst subarray to be removed to make array sorted
def findlengthofshortestsubarray(arr):
    n,left=len(arr),0
    while left+1<n and arr[left]<=arr[left+1]:
        left+=1
    if left==n-1:
        return 0
    right=n-1
    while right>0 and arr[right-1]<=arr[right]:
        right-=1
    result=min(n-left-1,right)

#34-longest repeating substring
def longestrepeatingsubstring(s):
    n=len(s)
    dp=[[0]*n for _ in range(n)]
    ans=0
    for i in range(n):
        for j in range(i+1,n):
            if s[i]==s[j]:
                dp[i][j]=dp[i-1][j-1]+1 if i else 1
                ans=max(ans,dp[i][j])
    return ans
#35-replace all ? to avoid consequative repeating charcter
def modifystring(s):
    char_list=list(s)
    n=len(s)
    for i in range(n):
        if char_list[i]=="?":
            for c in "abc":
                if (i>0 and char_list[i-1]==c)or (i+1<n and char_list[i+1]==c):
                    continue
            char_list[i]=c
            break
    return "".join(char_list)

#36-the length of longest substring without repeating charcter
def lengthoflongestsubstring(s):
    if len(s)==len(set(s)): return len(s)
    substring,maxlen="",0
    for i in s:
        if i not in substring:
            substring=substring+i
            maxlen=max(maxlen,len(substring))
        else:
            substring=substring.split(i)[1]+i
    return maxlen

#37-maximum repeating substring
def maxrepeating(sequence,word):
    if len(sequence)<len(word): return 0
    ans=0
    k=1
    while word*k in sequence:
        ans+=1
        k+=1
    return ans

#38-count sunstring without repeating charcter
def count_substring(s):
    count=0
    for i in range(len(s)):
        for j in range(i+1,len(s)):
            substring=s[i:j+1]
            if len(substring)==len(set(substring)):
                count+=1
    return count

#39-longest uncommon subsequence
def longestuncommon(a,b):
    if a==b: return -1
    else:
        return max(len(a),len(b))

#40-split string to max number unique string
def maxunique(s):
    global d
    maxm=0
    for i in range(1,len(s)+1):
        tmp=s[0:i]
        if tmp not in d:
            d[tmp]=1
            maxm=max(maxm,maxunique(s[i:])+1)
            del d[tmp]
    return maxm

#41- magical string
def magicalstring(n):
    s=["1","2","2"]
    for i in range(2,n):
        add_two=s[-1]=='-1'
        s.extend(list(int(s[i])*("2" if add_two else "1")))
        if len(s)>=n: break
    return s[:n].count('1')

#42- merge sorted array
def merge(nums1,m,nums2,n):
    t=m+n
    keyy=[]
    for i in (nums1[:m]):
        keyy.append(i)
    for i in (nums2[:n]):
        keyy.append(i)
    keyy.sort()
    nums1[:t]=keyy[:t]

#43-findthe distance value
def findthedistance(arr1,arr2,d):
    s=0
    for i in arr1:
        t=0
        for j in arr2:
            if abs(i-j)<=d:
                t=1
                break
        if t==0:
            s=s+1
    return s

#44- mergee intervals
def merge(intervals):
    if len(intervals)<=1: return intervals
    intervals.sort()
    res=[intervals[0]]
    for start,end in intervals[1:]:
        if start<=res[-1][-1]:
            res[-1][1]=max(end,res[-1][1])
        else:
            res.append([start,end])
    return res

#45-insert intervals
def insert(intervals,newintervals):
    s,e=newintervals.start,newintervals.end
    left,right=[],[]
    for i in intervals:
        if i.end<s:
            left+=i
        elif i.start>e:
            right+=i
        else:
            s=min(s,i.start)
            e=max(e,i.end)
    return left+[intervals(s,e)]+right

#46-none overlapping intervals
def erase_overlaping_intervals(intervals):
    intervals.sort()
    ans=0
    prevend=intervals[0][1]
    for start,end in intervals[1:]:
        if start>=prevend:
            prevend=end
        else:
            ans+=1
            prevend=min(end,prevend)
    return ans

#47-maximum points you can obtain from cards
def maxcore(cardpoints,k):
    dparray=[0 for i in range(k+1)]
    dparray[0]=sum(cardpoints[:k])
    for i in range(1,k+1):
        dparray[i]=dparray[i-1]-cardpoints[k-i]+cardpoints[-i]
    return max(dparray)

#48-Maximum Erasure Value
def maximumuniquesubarray(nums):
    s,suml,start,m=set(),0,0,0
    for i in range(len(nums)):
        while nums[i] in s:
            s.remove(nums[start])
            suml-=nums[start]
            start+=1
        s.add(nums[i])
        suml+=nums[i]
        m=max(m,suml)
    return m

#49 - number of substring containg all 3 chrcter
def numberofsubstring(s):
    l={'a':-1,'b':-1,'c':-1}
    ans=0
    for i,ch in enumerate(s):
        l[ch]=i
        ans+=min(l.values())+1
    return ans
#50-square of sorted array
def sortedsquares(nums):
    nums=[i*i for i in nums]
    nums.sort()
    return nums

#51-max consequative ones
def longestones(nums,K):
    if not nums : return 0
    max_0,c=0,0
    for i in nums:
        if i ==1:
            c+=1
        else:
            max_o=max(c,max_o)
            c=0
    return max(c,max_o)

#52-max consequative one 
def findmaxconsequativeones(nums):
    max_count=0
    count=0
    for i in nums:
        if i==1:
            count+=1
        else:
            max_count=max(max_count,count)
            count=0
    return max(max_count,count)

#53-Equilibrium Point
def findEquilibrium(arr):
    #code
    total_sum=sum(arr[1:])
    left_sum=0
    for i in range(1,len(arr)):
        total_sum-=arr[i]
        left_sum+=arr[i-1]
        if total_sum==left_sum:
            return i
    return -1
#####################################################################
#54-find peak element
def findPeakElement(nums):
    # Initialize the start and end pointers.
    start, end = 0, len(nums) - 1
    # Binary search to find the peak element.
    while start < end:
    # Find the middle index.
        mid = (start + end) // 2
        # If the middle element is greater than its next element,
        # it means a peak element is on the left side(inclusive of mid).
        if nums[mid] > nums[mid + 1]:
            end = mid
            # Otherwise, the peak is in the right half of the array.
        else:
            start = mid + 1
        # When start and end pointers meet, we've found a peak element.
    return start

#55-minimum distance between two distinct charcter in the arr
def minDist(arr, n, x, y):
    if x not in arr or y not in arr:return -1
    mn=float('inf')
    for i in range(len(arr)):
        if arr[i]==x:
            for j in range(len(arr)):
                if arr[j]==y:
                    mn=min(mn,abs(i-j))
    return mn

#56-first and last occurance of element in the arr
def firstAndLast(x, arr):
    c=[]
    for i in range(len(arr)):
        if arr[i]==x:
            c.append(i)
        if len(c)>=1:
            return c[0],c[-1]
        else:
            return [-1]

#57-Form a palindrome
#Given a string, find the minimum number of characters to be inserted to convert it to a palindrome.
def countMin(s):
    n = len(s)
    dp = [[0] * (n + 1) for _ in range(n + 1)]

    for i in range(n):
        for j in range(n):
            if s[i] == s[n - 1 - j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

    return n - dp[n][n]
    

#58--Remove all continuous occurrences of ‘a’ and all occurrences of ‘b’
'''
Input: str = “abcddabcddddabbbaaaaaa”
Output: acddacdddda
‘abcddabcddddabbbaaaaaa’ will not result in ‘acddacddddaa’ because after removing the required occurrences, the string will become ‘acddacddddaa’ which will result in ‘acddacdddda’

Input: str = “aacbccdbsssaba”
Output: acccdsssa
'''
def removeOccurrences(str) :
    # String to store the resultant string
    res = ""
    for i in range(len(str)) :
        # If 'a' appeared more than once continuously
        if (res) :
            if (str[i] == 'a' and res[-1] == 'a') :
                # Ignore the character
                continue
            # Ignore all 'b' characters
            elif (str[i] == 'b') :
                continue
            else :
                # Characters that will be included in the resultant string
                res += str[i]
        else :
            if (str[i] == 'a' ) :
                res += str[i]
            # Ignore all 'b' characters
            elif (str[i] == 'b') :
                continue
            else :
                # Characters that will be included  in the resultant string
                res += str[i]
    return res

#59--Best Time to Buy and Sell Stocks II
def maxProfit(A):
    sum = 0
    n = len(A)
    for i in range(1, n):
        if A[i - 1] - A[i] < 0:
            sum += (A[i] - A[i - 1])
    return sum
##############################################################################################
#60-Intersection Of Sorted Arrays
def intersect(A, B):
    ans = []
    i = 0
    j = 0
    while i < len(A) and j < len(B):
        if A[i] == B[j]:
            ans.append(A[i])
            i += 1
            j += 1
        elif A[i] > B[j]:
            j += 1
        else:
            i += 1
    return ans
############################################################################
#61-remove duplicate from sorted array
def removeDuplicates(nums):
    if not nums:
        return 0

    unique_index = 1

    for i in range(1, len(nums)):
        if nums[i] != nums[i - 1]:
            nums[unique_index] = nums[i]
            unique_index += 1

    return unique_index

#62-#Find all pairs with a given sum
'''
Input: x = 9, arr1[] = [1, 2, 4, 5, 7], arr2[] = [5, 6, 3, 4, 8]
Output:
1 8
4 5
5 4
Explanation: (1, 8), (4, 5), (5, 4) are the pairs which sum to 9.
'''
def allPairs(x, arr1, arr2):
    pairs = []
    seen = set(arr2)  # Store elements of arr2 in a set for fast lookup
    for num1 in arr1:
        complement = x - num1
        if complement in seen:
            pairs.append((num1, complement))

        pairs.sort(key=lambda pair: pair[0])

    return pairs

#63-Longest repeating subsequence
def LongestRepeatingSubsequence(str):
   memo = [[0 for j in range(len(str)+1)]for i in range(len(str)+1)]
   for i in range(1, len(memo)):
      for j in range(1,len(memo[0])):
          if str[i-1] == str[j-1] and i != j:
              memo[i][j] = memo[i-1][j-1]+1
          else:
              memo[i][j] = max(memo[i][j-1],memo[i-1][j])
   return memo[-1][-1]

#64-longest subsequence with limit sum
def answerQueries(nums, queries):
    numsSorted = sorted(nums)
    res = []
    for q in queries:
        total = 0
        count = 0
        for num in numsSorted:
            total += num
            count += 1
            if total > q:
                count -= 1
                break
        res.append(count)

    return res

#65-#Pair with given sum in a sorted array
'''
Input: k = 8, arr[] = [1, 2, 3, 4, 5, 6, 7]
Output: 3
Explanation: There are 3 pairs which sum up to 8 : {1, 7}, {2, 6}, {3, 5}
'''
def countPair (self, k, arr) :
    c=set()
    s=0
    for i in arr:
        if k-i in c:
            s+=1
        c.add(i)
    return s

#66-twoSum should return indices of the two numbers such that they add up to the target
def twoSum(A, B):
    numMap = {}  # Using a dictionary for the hash map

    for i in range(len(A)):
        complement = B - A[i]
        if complement in numMap:
            return [numMap[complement], i + 1]
        #Store the element only once in the dictionary to ensure the indices returned are the first and last occurrence.
        if A[i] not in numMap:
            numMap[A[i]] = i + 1

    return []
    
#67-Jump Game                             ###################################
'''
Input: arr[] = {1, 3, 5, 8, 9, 2, 6, 7, 6, 8, 9}
Output: 3
Explanation:First jump from 1st element to 2nd element with value 3. From here we jump to 5th element with value 9,
and from here we will jump to the last.
'''
class Solution(object):
    def jump(self, nums):
        # Initialize the jump count, the maximum reach, and the edge of the current range to 0.
        jump_count = max_reach = last_reach = 0

        # # Step 2: Iterate over the array except the last element.
        for index, value in enumerate(nums[:-1]):
            # Update the maximum reach with the furthest position we can get to from the current index.
            max_reach = max(max_reach, index + value)

            # If we have reached the furthest point to which we had jumped previously,
            # Increment the jump count and update the last reached position to the current max_reach.
            if last_reach == index:
                jump_count += 1
                last_reach = max_reach

        # Return the minimum number of jumps needed to reach the end of the list.
        return jump_count

#################################################################################################
#68-Jump Game ll       return True of False
'''
You are given an integer array nums. You are initially positioned at the array's first index,
and each element in the array represents your maximum jump length at that position.
Return true if you can reach the last index, or false otherwise.
Example 1:

Input: nums = [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
'''
def can_jum(arr):
    max_index=0
    for i,v in enumerate(arr):
        if i>max_index:return False
        max_index=max(max_index,i+v)
    return True

#69-Array Leaders
'''
An element of the array is considered a leader if it is greater than all the elements on its right side or if it is equal to the maximum element
on its right side. The rightmost element is always a leader.
Examples
Input: n = 6, arr[] = {16,17,4,3,5,2}
Output: 17 5 2
Explanation: Note that there is nothing greater on the right side of 17, 5 and, 2.
'''
def printLeaders(arr, n):
    ans = []

    # Last element of an array is always a leader,
    # push into ans array.
    max_elem = arr[n - 1]
    ans.append(arr[n - 1])

    # Start checking from the end whether a number is greater
    # than max no. from right, hence leader.
    for i in range(n - 2, -1, -1):
        if arr[i] > max_elem:
            ans.append(arr[i])
            max_elem = arr[i]
    return ans

#70-check if array is sorted
def is_sorted(arr):
  for i in range(len(arr) - 1):
    if arr[i] > arr[i + 1]:
      return False
  return True

#71-Count Complete Subarrays in an Array
'''
A subarray is considered "complete" if it contains exactly the same distinct elements as are present in the entire array.
To clarify, a subarray is a contiguous sequence of elements within the array.
The main goal is to find the number of such unique complete subarrays.

Input: nums = [1,3,1,2,2]
Output: 4
Explanation: The complete subarrays are the following: [1,3,1,2], [1,3,1,2,2], [3,1,2] and [3,1,2,2].
'''
def countCompleteSubarrays( nums):
    cnt = len(set(nums))   #determine the total number of distinct elements in the entire array nums.
    ans, n = 0, len(nums)   #0,5
    for i in range(n):
        s = set()
        for x in nums[i:]:
            s.add(x)
            if len(s) == cnt:
                ans += 1
    return ans
    
#72move zeros to the end of the array
def pushZerosToEnd(arr):
    # Pointer to track the position for next non-zero element
    count = 0
    for i in range(0,len(arr)):
        if arr[i]!=0:
            arr[count],arr[i]=arr[i],arr[count]
            count+=1
#############################################################################
#73-Rearrange array such that even positioned are greater than odd
'''
Input: N = 4, arr[] = {1, 2, 2, 1}
Output: 1 2 1 2
arr[i] >= arr[i-1], if i is even.
arr[i] <= arr[i-1], if i is odd.
'''
def rearrange(arr):
    N = len(arr)
    for i in range(1, N):
        # Check if the index is even (1-based) => i+1 is even
        if (i + 1) % 2 == 0:
            # Ensure arr[i] >= arr[i-1]
            if arr[i] < arr[i - 1]:
                arr[i], arr[i - 1] = arr[i - 1], arr[i]
        else:
            # Ensure arr[i] <= arr[i-1]
            if arr[i] > arr[i - 1]:
                arr[i], arr[i - 1] = arr[i - 1], arr[i]

#74-Count the number of possible triangles
def squared(arr):
    squared=set()
    for i in arr:
        squared.add(arr[i]*arr[i])
    for i in range(len(arr)):
        for j in range(i+1,len(arr)):
            if arr[i]*arr[i]+arr[j]*arr[j] in squared:
                count+=1
    return count

#75-Remove Duplicates from Sorted Array II
'''
Given nums = [1,1,1,2,2,3],
Your function should return length = 5, with the first five elements of nums being 1, 1, 2, 2 and 3 respectively.
It doesn't matter what you leave beyond the returned length
'''
def removeDuplicates(nums):
    if len(nums) <= 2:
        return len(nums)
    slow = 2
    for fast in range(2, len(nums)):
        if nums[fast] == nums[slow-1] and nums[fast] == nums[slow-2]:
            continue
        nums[slow] = nums[fast]
        slow += 1

    return slow

#76-Reduce the array such that each element appears at most K times
'''
Input: arr[] = {1, 2, 2, 2, 3}, K = 2
Output: {1, 2, 2, 3}
Explanation:
Remove 2 once, as it occurs more than 2 times.
'''
from collections import Counter
def reduceArray(arr, n, K) :
    freq=Counter(arr)
    ans=[]
    for i in range(n):
        if freq[arr[i]]>=2:
            freq[arr[i]]-=1
            ans.append(arr[i])
        if freq[arr[i]]==1:
            freq[arr[i]]=0
            ans.append(arr[i])
    return ans

#77-largest-element-in-an-array-after-merge-operations
'''
Example 1:
Input: nums = [2,3,7,9,3]
Output: 21
Explanation: We can apply the following operations on the array:
- Choose i = 0. The resulting array will be nums = [5,7,9,3].
- Choose i = 1. The resulting array will be nums = [5,16,3].
- Choose i = 0. The resulting array will be nums = [21,3].
The largest element in the final array is 21. It can be shown that we cannot obtain a larger element.
'''

def maxArrayValue(nums):
    # Loop through the array in reverse order except for the last element.
    for i in range(len(nums) - 2, -1, -1):
        # If the current element is less than or equal to the next one,
        # modify the current element by adding the next element's value to it.
        if nums[i] <= nums[i + 1]:
            nums[i] += nums[i + 1]

        # After modifying the array, return the maximum value in the array.
    return max(nums)
    
#78-Make all Ones together by Shifting Ones.
'''
Input: 11011
Output: 2
Explanation:  In the first operation move ‘1’  at index 3 to index 2. Now the string becomes 11101.
In the second operation move ‘1’ at index 4 to index 3. Now the string becomes 11110. Therefore, the answer is 2.
'''
def make_all_ones_together(s):
    n=len(s)
    count_one=s.count('1')
    ans=0
    left_one=0
    for i in range(n):
        if s[i]=='0':
            ans+=min(count_one-left_one,left_one)
        if s[i]=='1':
            left_one+=1
    return ans
##############################################################################################
#79-Find four elements a, b, c and d in an array such that a+b = c+d
def find_pairs(nums):
    pair_sums = {}
    result = []
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            sum_ij = nums[i] + nums[j]
            if sum_ij in pair_sums:
                result.append([pair_sums[sum_ij][0], pair_sums[sum_ij][1], nums[i], nums[j]])
            else:
                pair_sums[sum_ij] = [nums[i], nums[j]]
    return result

#80-Maximum distance between two occurrences of same element in array
def max_distance(arr):
    n=len(arr)
    hashmap={}
    max_distance=0
    for i in range(n):
        if arr[i] not in hashmap.keys():
            hashmap[arr[i]]=i
        else:
            max_distance=max(max_distance,i-hashmap[arr[i]])
    return max_distance
#############################################################################################
#81-Rearrange an array such that arr[i] = i
'''
Input : arr = {-1, -1, 6, 1, 9, 3, 2, -1, 4, -1}
Output : [-1, 1, 2, 3, 4, -1, 6, -1, -1, 9]
'''
def fixArray(arr, n):
    s=set() #-1,6,1,9,3,2,4
    for i in range(len(arr)):
        s.add(arr[i])
    for i in s:
        arr[i]=i    #-1,1,2,3,4,-1,6,-1,-
    else:
        arr[i]=-1
    return arr

#82- product of array except itself
def productExceptSelf(nums):
    n = len(nums)
    postfix=[1]*n
    prefix=[1]*n
    result=[0]*n
    for i in range(1,n):
        prefix[i]=prefix[i-1]*nums[i-1]
    for i in range(n-2,-1,-1):
        postfix[i]=postfix[i+1]*nums[i+1]
    for i in range(n):
        result[i]=prefix[i]*postfix[i]
    return result

#83-Three Sum Smaller
'''
Problem Statement: Given an array of n integers nums and an integer target, find the number of index triplets i, j, k
with 0 <= i < j < k < n that satisfy the condition nums[i] + nums[j] + nums[k] < target.
'''
def threeSumSmaller(nums, target):
    ans,n=0,len(nums)
    nums.sort()
    for i in range(n-2):
        l,r=i+1,n-1
        while r<l:
            curr_sum=nums[i]+nums[r]+nums[l]
            if curr_sum<target:
                ans+=r-l
                l+=1
            else:
                r-=1
    return ans
######################################################################################
#84-Two Sum I
#Problem Statement: Given an array of integers nums and an integer target, return indices of the two numbers
#such that they add up to target.

def twoSum(nums, target):
    hashmap={}
    for i,num in enumerate(nums):
        desired=target-num
        if desired in hashmap:
            return [hashmap[desired],i]
        hashmap[num]=i
    return []
 ###################################################################################
# 85-Two Sum Less Than K
'''
Problem Statement: Given an array nums of integers and integer k, return the maximum sum such that there exists
i < j with nums[i] + nums[j] = sum and sum < k. If no i, j exist satisfying this equation, return -1.
'''
#Given an array A of integers and integer K, return the maximum S such that there exists i < j with A[i] + A[j] = S and S < K.
#If no i, j exist satisfying this equation, return -1.

def twoSumLessThanK(A, k):
    seen=set()
    for num in A:
        if k-num in seen:
            return num+(k-num)
        seen.add(num)
    return -1
##################################################################################
#86-Three Sum
'''
Problem Statement: Given an array and a value, return a triplet whose sum is equal to the given target otherwise return []
'''
def find_triplets(nums, target):
    n=len(nums)
    nums.sort()
    for i in range(n):
        r,l=i+1,n-1
        while r<l:
            curr_sum=nums[i]+nums[r]+nums[l]
            if curr_sum==target:
                return [nums[i],nums[r],nums[l]]
            elif curr_sum>target:
                l+1
            else:
                r-1
    return []

#######################################################################################
#87-Three Sum Zero
########################################
def threeSum(nums):
    n=len(nums)
    nums.sort()
    for i in range(n):
        r,l=i+1,n-1
        while r<l:
            curr_sum=nums[i]+nums[l]+nums[r]
            if curr_sum==0:return[nums[i],nums[r],nums[l]]
            elif curr_sum>0:l+1
            else:r-1
    return []
    
#88- max consequitve one I  max consequative charcters
from itertools import groupby
'''
Given a binary array nums, return the maximum number of consecutive 1's in the array.
Input: nums = [1,1,0,1,1,1]
Output: 3
Explanation: The first two digits or the last three digits are consecutive 1s. The maximum number of consecutive 1s is 3.
'''
def findMaxConsecutiveOnes(nums):
    return max((sum(g) for k, g in groupby(nums) if k == 1), default=0)


#89-Maximum consecutive one’s (or zeros) in a binary array
'''
Given a binary array, find the count of a maximum number of consecutive 1s present in the array.
'''
def getMaxLength(arr, n):
    # initialize count
    count = 0
    # initialize max
    result = 0
    for i in range(0, n):
        # Reset count when 0 is found
        if (arr[i] == 0):count = 0
        # If 1 is found, increment count# and update result if count # becomes more.
        else:
            # increase count
            count+= 1
            result = max(result, count)
    return result

#90-first and last occurance of element in the arr
def firstAndLast(self, x, arr):
    c=[]
    for i in range(len(arr)):
        if arr[i]==x:
            c.append(i)
        if len(c)>=1:
            return c[0],c[-1]
        else:
            return [-1]
        

#91-Pair With Given Difference
'''
Input 1:
 A = [5, 10, 3, 2, 50, 80]  B = 78
Input 2:A = [-10, 20]       B = 30
Example Output
Output 1: 1
Output 2: 1
'''
def solve(A, B):
    """
    Given an array of integers A and an integer B, checks if there exists 2 numbers
    in A whose sum or difference is B.
    For example:
    solve([5, 10, 3, 2, 50, 80], 78) == 1
    solve([5, 10, 3, 2, 50, 80], 70) == 0
    """
    s = set()
    for num in A:
        if num + B in s or num - B in s:
            return 1
        s.add(num)
    return 0
