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

#include "main_window.h"

#include "base/base_impl.h"
#include <QtGui>

static constexpr int kQtBorder = 11 * 2; // 11 for default qt margin | *2 for both sides.

MainWindow::MainWindow(const std::string& stream_name, const bool slider)
  : stream_name_(stream_name),
    use_slider_(slider),
    mouse_point_(QPoint(0, 0)),
    space_key_pressed_(false) {
  // Create simple widget used for display.
  main_widget_ = new QLabel(this);
  display_widget_ = new QLabel(this);
  setCentralWidget(main_widget_);
  setWindowTitle(QString(stream_name.c_str()));

  // Hierarchy level slider.
  max_slider_level_ = 20;
  curr_slider_level_  = max_slider_level_ / 2;
  hierarchy_slider_ = new QSlider(Qt::Horizontal);
  hierarchy_slider_->setMinimum(0);
  hierarchy_slider_->setMaximum(max_slider_level_);
  hierarchy_slider_->setTickPosition(QSlider::TicksBelow);
  hierarchy_slider_->setValue(curr_slider_level_);
  connect(hierarchy_slider_, SIGNAL(sliderMoved(int)), this, SLOT(ChangeLevel(int)));

  // GUI layout.
  QGridLayout* centralLayout = new QGridLayout;
  centralLayout->addWidget(display_widget_);
  if (use_slider_) {
    centralLayout->addWidget(hierarchy_slider_);
  }

  main_widget_->setLayout(centralLayout);
}

void MainWindow::SetSize(int sx, int sy) {
  display_widget_->resize(sx, sy);
  const int room_for_slider = use_slider_ ? 3 * kQtBorder : kQtBorder;
  main_widget_->resize(sx + kQtBorder, sy + room_for_slider);
  resize(sx + kQtBorder, sy + room_for_slider);
}

void MainWindow::DrawImage(const QImage& image) {
  display_widget_->setPixmap(QPixmap::fromImage(image));
  display_widget_->update();
}

void MainWindow::ChangeLevel(int n) {
  curr_slider_level_ = n;
  hierarchy_slider_->setValue(curr_slider_level_);
}

float MainWindow::GetLevel() const {
  return curr_slider_level_ * (1.0f / max_slider_level_);
}

void MainWindow::SetLevel(float level) {
  CHECK_LE(level, 1.0);
  CHECK_GT(level, 0.0);
  ChangeLevel(level * max_slider_level_ + 0.5f);
}

void MainWindow::mousePressEvent(QMouseEvent* event) {
  if (event->button() == Qt::LeftButton) {
    mouse_point_ = event->pos();
    LOG(INFO) << "Selected point: " << mouse_point_.x() << ", " << mouse_point_.y();
  }
}

void MainWindow::keyPressEvent(QKeyEvent *event) {
  if (event->key() == Qt::Key_Escape) {
    LOG(INFO) << "Shutting down.";
    exit(0);
  }
  if (event->key() == Qt::Key_Space) {
    space_key_pressed_ = true;
  }
}

std::pair<int, int> MainWindow::GetMouseLoc() {
  return std::make_pair<int, int>(mouse_point_.x() - kQtBorder / 2,
                                  mouse_point_.y() - kQtBorder / 2);
}

bool MainWindow::SpaceKeyPressed() {
  bool ret = space_key_pressed_;
  space_key_pressed_ = false;
  return ret;
}
