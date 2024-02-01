graph [
  node [
    id 0
    label "5"
    gate "H"
    shape 2
    shape 2
  ]
  node [
    id 1
    label "6"
    gate "CX_0"
    shape 2
    shape 2
    shape 2
  ]
  node [
    id 2
    label "7"
    gate "CX_1"
    shape 2
    shape 2
    shape 2
  ]
  node [
    id 3
    label "8"
    gate "CX_0"
    shape 2
    shape 2
    shape 2
  ]
  node [
    id 4
    label "9"
    gate "CX_1"
    shape 2
    shape 2
    shape 2
  ]
  node [
    id 5
    label "10"
    gate "CX_0"
    shape 2
    shape 2
    shape 2
  ]
  node [
    id 6
    label "11"
    gate "CX_1"
    shape 2
    shape 2
    shape 2
  ]
  node [
    id 7
    label "12"
    gate "CX_0"
    shape 2
    shape 2
    shape 2
  ]
  node [
    id 8
    label "13"
    gate "CX_1"
    shape 2
    shape 2
    shape 2
  ]
  node [
    id 9
    label "14"
    gate "CX_0"
    shape 2
    shape 2
    shape 2
  ]
  node [
    id 10
    label "15"
    gate "CX_1"
    shape 2
    shape 2
    shape 2
  ]
  node [
    id 11
    label "16"
    gate "CX_0"
    shape 2
    shape 2
    shape 2
  ]
  node [
    id 12
    label "17"
    gate "CX_1"
    shape 2
    shape 2
    shape 2
  ]
  node [
    id 13
    label "18"
    gate "CX_0"
    shape 2
    shape 2
    shape 2
  ]
  node [
    id 14
    label "19"
    gate "CX_1"
    shape 2
    shape 2
    shape 2
  ]
  node [
    id 15
    label "20"
    gate "CX_0"
    shape 2
    shape 2
    shape 2
  ]
  node [
    id 16
    label "21"
    gate "CX_1"
    shape 2
    shape 2
    shape 2
  ]
  node [
    id 17
    label "22"
    gate "RZ"
    shape 2
    shape 2
  ]
  node [
    id 18
    label "23"
    gate "S"
    shape 2
    shape 2
  ]
  node [
    id 19
    label "24"
    gate "H"
    shape 2
    shape 2
  ]
  node [
    id 20
    label "25"
    gate "S"
    shape 2
    shape 2
  ]
  node [
    id 21
    label "26"
    gate "RZ"
    shape 2
    shape 2
  ]
  edge [
    source 0
    target 1
    rgreedy 5
  ]
  edge [
    source 1
    target 2
    rgreedy 9
  ]
  edge [
    source 1
    target 15
    rgreedy 6
  ]
  edge [
    source 2
    target 3
    rgreedy 8
  ]
  edge [
    source 3
    target 4
    rgreedy 7
  ]
  edge [
    source 3
    target 13
    rgreedy 12
  ]
  edge [
    source 4
    target 5
    rgreedy 20
  ]
  edge [
    source 5
    target 6
    rgreedy 17
  ]
  edge [
    source 5
    target 11
    rgreedy 14
  ]
  edge [
    source 6
    target 7
    rgreedy 16
  ]
  edge [
    source 7
    target 8
    rgreedy 19
  ]
  edge [
    source 7
    target 9
    rgreedy 17
  ]
  edge [
    source 8
    target 10
    rgreedy 18
  ]
  edge [
    source 9
    target 10
    rgreedy 19
  ]
  edge [
    source 9
    target 12
    rgreedy 13
  ]
  edge [
    source 11
    target 12
    rgreedy 15
  ]
  edge [
    source 11
    target 14
    rgreedy 20
  ]
  edge [
    source 13
    target 14
    rgreedy 10
  ]
  edge [
    source 13
    target 16
    rgreedy 11
  ]
  edge [
    source 15
    target 16
    rgreedy 12
  ]
  edge [
    source 15
    target 17
    rgreedy 2
  ]
  edge [
    source 17
    target 18
    rgreedy 3
  ]
  edge [
    source 18
    target 19
    rgreedy 1
  ]
  edge [
    source 19
    target 20
    rgreedy 0
  ]
  edge [
    source 20
    target 21
    rgreedy 4
  ]
]
