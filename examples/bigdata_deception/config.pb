language: PYTHON
name: "bigdata_deception"

variable {
 name: "SIZE"
 type: INT
 size: 1
 min: 1
 max: 150 #approx. e^5
}

variable {
 name: "X"
 type: FLOAT
 size: 2
 min: 0
 max: 1
}

# Integer example
#
# variable {
# name: "Y"
# type: INT
# size: 5
# min: -5
# max: 5
# }

# Enumeration example
#
# variable {
# name: "Z"
# type: ENUM
# size: 3
# options: "foo"
# options: "bar"
# options: "baz"
# }


