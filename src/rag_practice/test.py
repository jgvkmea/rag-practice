# 1. 何のコードファイルかを示すdocstringがファイルの最初にない
# 2. ソートされておらず、不要なimport文がある
import math
import random

my_list = [1, 2, 3, 4, 5]  # 3. スペースのないリスト


# 4. アノテーション(型)定義をしていない
def example_function(x, y):
    # 5. 関数の説明がない
    random_value = random.randint(1, 10)
    sum_of_list = sum(my_list)
    result = x + y + sum_of_list + math.sqrt(random_value)
    return result
