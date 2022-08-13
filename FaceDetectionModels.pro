TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

CONFIG += link_pkgconfig
PKGCONFIG += opencv4

LIBS += -ldlib
PKGCONFIG += dlib-1
LIBS += -lblas
LIBS += -llapack

SOURCES += \
        main.cpp
