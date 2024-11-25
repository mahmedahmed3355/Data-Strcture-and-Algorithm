#1-fractional knapsack
def fractional_knapsack(w,arr,n):
    arr.sort(key=lambda x:x.value/x.weight,reverse=True)
    curr_weight,final_value=0,0.0
    for i in range(n):
        if curr_weight+arr[i].weight<=w:
            curr_weight+=arr[i].weight
            final_value+=arr[i].value
        else:
            remain=w-curr_weight
            final_value+=arr[i].value/arr[i].weight*remain
            break
    return final_value
#2-target sum
def target_sum(nums,target):
    total_sum=sum(nums)
    #check the error
    if target>total_sum or (total_sum-target)%2 !=0:
        return 0
    subset_sum=(total_sum-target)//2
    dp=[0]*subset_sum+1
    dp[0]=1
    for num in nums:
        for j in range(subset_sum,nums-1,-1):
            dp[j]+=dp[j-num]
    return dp[-1]
#3-climbing_staris
def climbing_stairs(n):
    if n==1 or n==2 :
        return n
    a,b=1,2
    for _ in range(3,n+1):
        a,b=b,a+b
    return b
#4-min cost climbing stair
def mincostclimbingstairs(cost):
    n=len(cost)
    dp=[0]*n
    dp[0],dp[1]=cost[0],cost[1]
    for i in range(2,n):
        dp[i]=cost[i]+min(dp[i-1],dp[i-2])
    return min(dp[n-1],dp[n-2])
#5-breaking integer to get max product
def max_prod(n):
    if n==2 or n==3 : return n-1
    res=1
    while n>4:
        n-=3
        res*=3
    return (n*res)
#6-coin changes
import sys
def count(coins,n,sum):
    if sum==0: return 1
    if sum<0: return 0
    if n<=0 : return 0
    return count(coins,n-1,sum)+count(coins,n,sum-coins[n-1])
#7-minimum coins to make sum
def mincoins(coins,m,sum):
    if sum==0 :return 0
    res=sys.maxsize
    for i in range(0,m):
        if (coins[i]<=sum):
            sub_res=mincoins(coins,m,sum-coins[i])

#8-partition a set into two subset of equal sum
def canpartition(nums):
    total_sum=sum(nums)
    if total_sum%2:
        return False
    subset_sum=total_sum//2
    dp=[False]*(subset_sum+1)
    dp[0]=True
    for num in nums:
        for _sum in reversed(range(num,subset_sum+1)):
            dp[_sum]=dp[sum]or dp[_sum-num]
            if dp[subset_sum]:
                return dp[subset_sum]
    return dp[subset_sum]
#9-count of subset with sum equal to X
from itertools import combinations
def count_subset_tab(arr,k):
    return sum(1 for i in range(len(arr)+1) for subset in combinations(arr,i)if sum(subset)==k)
#10-maximum subarray
def maxsubarray(nums):
    current_sum=nums[0]
    max_sum=nums[0]
    for i in range(1,len(nums)):
        current_sum=max(nums[i],current_sum+nums[i])
        max_sum=max(max_sum,current_sum)
    return max_sum
#11-edit distance
def edit_distance(string1,string2):
    if len(string1)>len(string2):
        diff=len(string1)-len(string2)
        string1[:diff]
    elif len(string2)>len(string1):
        diff=len(string2)-len(string1)
        string2[:diff]
    else:
        diff=0
    for i in range(len(string1)):
        if string1[i]!=string2[i]:
            diff+=1
    return diff
#12-rod cutting
def rodcutting(n,prices):
    val=[0 for x in range(n+1)]
    for i in range(1,n+1):
        max_val=float('-inf')
        for j in range(i):
            max_val=max(max_val,prices[j]+val[i-j-1])
        val[i]=max_val
    return val[n]
#13-maximum ribben cut
import math
def max_prod(n):
    if n==2 or n==1 : return n-1
    num_three=n//3
    num_two=0
    if n%3==1:
        num_three-=1
        num_two=2
    elif n%3==2:
        num_two=1
    res=1
    res*=math.pow(3,num_three)
    res*=math.pow(2,num_two)
    return int(res)
#14-max length by cutting n given wood into k pieces
def wood_cut(L,k):
    if sum(L)<k:return 0
    st,ed=1,max(L)
    while st+1<ed:
        md=st+(ed-st)/2
        pieces=sum([l/md for l in L])
        if pieces>=k:
            st=md
        else:
            ed=md
    if sum(l/ed for l in L)>=k:
        return ed
    return st
#15-rod cutting problem
def max_revenue(n,prices):
    dp=[0]*(n+1)
    for i in range(1,n+1):
        for j in range(1,i+1):
            dp[i]=max(dp[i],prices[j-1]+dp[i-j])
    return dp[n]
#16-split arr to maximum possible subset having product of their length with max element at leas k
def countofsubsets(arr,n,k):
    count,pre=0,1
    arr.sort()
    for i in range(n):
        if arr[i]*pre>=k:
            count+=1
            pre=1
        else:
            pre+=1
    return count
#17-find all factor of a natural number
def printdivisors(n):
    i=1
    while i<=n:
        if(n%i==0):
            print(i,end="")
        i=i+1

#18-word break1
def wordbreak(s,wordset):
    dp=[False]*(len(s)+1)
    dp[0]=True
    for i in range(1,len(s)+1):
        for w in wordset:
            if dp[i-len(w)]and s[i-len(w):i]==w:
                dp[i]=True
    return dp[-1]
#19-word break 2
def wordbreak(s,worddict):
    def dfs(start_index,path):
        if start_index==len(s):
            ans.append("".join(path))
            return
        for end_index in range(start_index,len(s)):
            w=s[start_index:end_index+1]
            for w in worddict:
                path.append(w)
                dfs(end_index+1,path)
                path.pop()
    ans=[]
    dfs(0,[])
    return ans
#20-house robber1
def maxlootrec(hval,n):
    if n<=0:return 0
    if n==1: return hval[0]
    pick=hval[n-1]+maxlootrec(hval,n-2)
    not_pick=maxlootrec(hval,n-1)
    return max(pick,not_pick)

#21-house robber2
def rob(nums):
    def _rob_subsequence(sub_nums):
        prev_max=curr_max=0
        for val in sub_nums:
            prev_max,curr_max=curr_max,max(curr_max,prev_max+val)
        return curr_max
    if len(nums)==1: return nums[0]
    return max(_rob_subsequence(nums[1:]),_rob_subsequence(nums[:-1]))
#22-jump Game
def can_jump(nums):
    n=len(nums)
    target=n-1
    for i in range(n-1,-1,-1):
        max_jump=nums[i]
        if i+max_jump>=target:
            target=i
    return target==0
#23-jump_Game2
def jump(nums):
    smallest,n,end,far=0,len(nums),0,0
    for i in range(n-1):
        far=max(far,i+nums[i])
        if i==end:
            smallest+=1
            end=far
    return smallest
#24-delete and earn
def deleteAndEarn(nums):
    if not nums : return 0
    house = [0] * (max(nums)+1)
    for num in nums:
        house[num] += num

    dp = [0]*(len(house)+1)
    for i in range(1,len(house)):
        dp[i] = max(house[i]+dp[i-2], dp[i-1])
    return max(dp[-1],dp[-2])
#25-longest commonsubsequence(s1,s2):
def get_lcs_length(s1,s2):
    m,n=len(s1),len(s2)
    dp=[[0]*(n+1)for x in range(m+1)]
    for i in range(1,m+1):
        for j in range(1,n+1):
            if s1[i-1]==s2[j-1]:
                dp[i][j]=dp[i-1][j-1]+1
            else:
                dp[i][j]=max(dp[i-1][j],dp[i][j-1])
    return dp[m][n]
#26-longest increasing subsequence
def lengthOfLIS(nums) :
    n=len(nums)
    arr=[1]*n
    for i in range(1,n):
        for j in range(i):
            if nums[i]>nums[j]:
                arr[i]=max(arr[i],arr[j]+1)
    return max(arr)    
#27-max_sum increasing subsequence
def maxsum(arr):
    n=len(arr)
    lis=[i for i in arr]
    for i in range(n):
        for j in range(i):
            if arr[i]>arr[j]:
                lis[i]=max(lis[i],arr[i]+lis[j])
    return max(lis)
#28-best team with no conflict
def best_teamscore(scores,ages):
    team=list(zip(ages,scores))
    team.sort()
    n=len(ages)
    dp=[0]*n
    for i in range(n):
        curr=team[i][1]
        dp[i]=curr
        for j in range(i):
            if team[j][1]<=curr:
                dp[i]=max(dp[i],dp[j]+curr)
        return max(dp)
#29-longest repeating subsequence
def findlongestreaptingsubseq(str):
    n=len(str)
    dp=[[0 for i in range (n+1) ]for j in range(n+1)]
    for i in range(1,n+1):
        for j in range(1,n+1):
            if (str[i-1]==str[j-1] and i!=j):
                dp[i][j]=1+dp[i-1][j-1]
            else:
                dp[i][j]=max(dp[i][j-1],dp[i-1][j])
    return dp[n][m]
#30-longest subsequence with limit sum
def answerqueries(nums,queries):
    numsorted=sorted(nums)
    res=[]
    for q in queries:
        total,count=0,0
        for num in numsorted:
            total+=num
            count+=1
            if total>q:
                count-=1
        res.append(count)
    return res
#31-longest arithmetic subsequence
def longestaritchment(nums):
    dp={}
    for r in range(len(nums)):
        for l in range(0,r):
            diff=nums[r]-nums[l]
            seq_length=dp.get((l,diff),1)
            seq_length+=1
            dp[(r,diff)]=seq_length
    return max(dp.values())
#32-longest uncommonsubsequence1
def findluslength(a,b):
    if a==b: return -1
    else:
        return max(len(a),len(b))

#34-best team with no conflits
def bestteamscore(ages,scores):
    team=list(zip(ages,scores))
    team.sort()
    n=len(ages)
    dp=[0]*n
    for i in range(n):
        curr=team[i][1]
        dp[i]=curr
        for j in range(i):
            if team[j][1]<=curr:
                dp[i]=max(dp[i],dp[i]+curr)
    return max(dp)
#34-longest uncommon subsequence2
def findluslength(strs):
    strs.sort(key=lambda s:(-len(s),s))
    for i,s in enumerate(strs):
        if i>0 and strs[i-1]==s or i<len(strs)-1 and strs[i+1]==s:
            continue
        for j in range(0,i):
            it=iter(0,i)
            if all(c in it for c in s):
                break
        else:
            return(len(s))
    return -1
#35-longest alternating subsequence
def alternatingsubsequence(arr):
    return sum(1 for a,b in zip(arr,arr[1:]) if a*b<0)+1
#36-split string into max number unique string
from collections import Counter
def numsplits(s):
    cnt=Counter(s)
    vis=set()
    ans=0
    for c in s:
        vis.add(c)
        cnt[c]-=1
        if cnt[c]==0:
            cnt.pop(c)
        ans+=len(vis)==len(cnt)
    return ans
#37-unique path
def uniquepaths(m,n):
    total=m+n
    choose=min(m,n)-1
    product=1
    for i in range(choose):
        product=product*total/(i+1)
        total-=1
    return int(product)
#38-unbound knapsack
def unbounded_knapsack(n, wt, val, maxWeight):
    dp = [0] * (maxWeight + 1)
    for i in range(n):
        for j in range(wt[i], maxWeight + 1):
            dp[j] = max(dp[j], val[i] + dp[j - wt[i]])
    return dp[maxWeight]
#39-Best Time to Buy and Sell Stock
def maxProfit(prices):
  if prices is None or len(prices) == 0:
    return 0

  local_profit = 0
  global_profit = 0

  for i in range(1, len(prices)):
    local_profit = max(0, local_profit - prices[i - 1] + prices[i])
    global_profit = max(global_profit, local_profit)

  return global_profit