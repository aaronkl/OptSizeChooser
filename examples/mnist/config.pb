language: PYTHON
name:     "svm_on_mnist"


#variable {
# name: "Size"
# type: FLOAT
# size: 1
# min:  1
# max:  4.602059991
#}

#variable {
# name: "Size"
# type: INT
# size: 1
# min:  2
# max:  40000
#}

#log2
variable {
 name: "C"
 type: FLOAT
 size: 1
 min:  -5
 max:  15
}

#-log2
#variable {
# name: "gamma"
# type: FLOAT
# size: 1
# min:  -15
# max:  3
#}

