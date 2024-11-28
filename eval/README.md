label和prediction各存一个文件，格式是\<NEWLINE\> | xx | xx | xx |

执行calc_metrics文件，目前是依赖于print的。_test_hyp.txt  _test_tgt.txt  --row-header  --col-header

前两个就是路径，hyp是prediction，tgt是label。--row-header  --col-header这两个是控制“是否认为行列存在header”，比如如果行不存在header，即相当于不存在表头，那每一个元素是由“行名+自身value”组成。反之每个元素由“列名+行名 +自身value”组成。其他类同
