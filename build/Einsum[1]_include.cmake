if(EXISTS "/home/yx/code/EasyEinsum/build/Einsum[1]_tests.cmake")
  include("/home/yx/code/EasyEinsum/build/Einsum[1]_tests.cmake")
else()
  add_test(Einsum_NOT_BUILT Einsum_NOT_BUILT)
endif()
