// Copyright (c) 2010-2014, The Video Segmentation Project
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the The Video Segmentation Project nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// ---

#ifndef VIDEO_SEGMENT_VIDEO_DISPLAY_QT_MAIN_WINDOW_H__
#define VIDEO_SEGMENT_VIDEO_DISPLAY_QT_MAIN_WINDOW_H__

#include "base/base.h"
#include <QMainWindow>

class QLabel;
class QImage;
class QSlider;
class QMouseEvent;
class QKeyEvent;

class MainWindow : public QMainWindow {
  Q_OBJECT

public:
  explicit MainWindow(const std::string& stream_name, const bool slider = false);
  void SetSize(int sx, int sy);
  void DrawImage(const QImage& image);

  // Returns percentage.
  float GetLevel() const;

  // Sets fractional level (in [0, 1]).
  void SetLevel(float level);

  // Mouse location.
  std::pair<int, int> GetMouseLoc();

  // Keyboard handler.
  bool SpaceKeyPressed();

public slots:
  void ChangeLevel(int);

protected:
  void mousePressEvent(QMouseEvent* event);
  void keyPressEvent(QKeyEvent* event);

private:
  QLabel* main_widget_;
  QLabel* display_widget_;
  QSlider* hierarchy_slider_;

  std::string stream_name_;
  bool use_slider_;
  int curr_slider_level_;
  int max_slider_level_;
  bool space_key_pressed_;

  QPoint mouse_point_;
};

#endif // VIDEO_SEGMENT_VIDEO_DISPLAY_QT_MAIN_WINDOW_H__
