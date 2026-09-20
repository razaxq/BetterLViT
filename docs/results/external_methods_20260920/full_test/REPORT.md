# MMI-UNet 官方权重完整 Test 结果

QaTa-COV19-v2 Test：2113张，图像＋对应文本，224px、24tokens、作者原预处理、原始单通道GT计分、阈值严格>0.5、无后处理、逐图macro。

- IoU：75.4783%
- Dice：83.9707%
- 用时：318.31秒，本地CPU，batch1。

使用之前同一官方cov19权重，SHA256 3aa0e90851ed496035a5fb4b9d576fcb27ad088d1d5fe6a228d98cbc51804a4f；614张量严格恢复并逐一相等验证。全部2113张图像及GT字节哈希与服务器GuideDecoder/DD-CMD测试数据一致。原八图概率数组及指标完全再现，未改变原图结果。保存TEST_RESULTS.json、per_image.csv、VERIFIED.json及服务器数据哈希清单。

官方权重训练文件清单仍未独立核验；该成绩为官方checkpoint在当前完整Test上的测量，不能据此确认无训练重叠，亦不能混称与另外两项完全相同训练协议的重训成绩。
