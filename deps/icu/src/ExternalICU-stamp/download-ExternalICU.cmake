if(EXISTS "/home/pranav/Documents/meta/deps/icu/src/icu4c-57_1-src.tgz")
  file("MD5" "/home/pranav/Documents/meta/deps/icu/src/icu4c-57_1-src.tgz" hash_value)
  if("x${hash_value}" STREQUAL "x976734806026a4ef8bdd17937c8898b9")
    return()
  endif()
endif()
message(STATUS "downloading...
     src='http://download.icu-project.org/files/icu4c/57.1/icu4c-57_1-src.tgz'
     dst='/home/pranav/Documents/meta/deps/icu/src/icu4c-57_1-src.tgz'
     timeout='none'")




file(DOWNLOAD
  "http://download.icu-project.org/files/icu4c/57.1/icu4c-57_1-src.tgz"
  "/home/pranav/Documents/meta/deps/icu/src/icu4c-57_1-src.tgz"
  SHOW_PROGRESS
  # no TIMEOUT
  STATUS status
  LOG log)

list(GET status 0 status_code)
list(GET status 1 status_string)

if(NOT status_code EQUAL 0)
  message(FATAL_ERROR "error: downloading 'http://download.icu-project.org/files/icu4c/57.1/icu4c-57_1-src.tgz' failed
  status_code: ${status_code}
  status_string: ${status_string}
  log: ${log}
")
endif()

message(STATUS "downloading... done")
