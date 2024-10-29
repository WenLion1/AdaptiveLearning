def signed_angle_diff(angle1, angle2):
    # 计算顺时针和逆时针的角度差
    diff = (angle2 - angle1) % 360  # 计算模360后的差值
    # 如果差值大于180，使用逆时针的差值
    if diff > 180:
        diff -= 360
    return diff

# 示例
angle1 = 400
angle2 = 10
result = signed_angle_diff(angle1, angle2)
print(f"The signed angle difference between {angle1}° and {angle2}° is {result}°.")
