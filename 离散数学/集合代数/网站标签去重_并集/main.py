user1 = {"java", "redis", "kafka"}
user2 = {"python", "redis", "vue"}
pool = user1 | user2          # 并集
print(pool)                   # {'kafka', 'java', 'python', 'redis', 'vue'}