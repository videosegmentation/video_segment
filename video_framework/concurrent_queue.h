// concurrent_queue based on Anthony Williams,
// http://www.justsoftwaresolutions.co.uk/threading/implementing-a-thread-safe-queue-using-condition-variables.html

#ifndef VIDEO_SEGMENT_VIDEO_FRAMEWORK_CONCURRENT_QUEUE_H__
#define VIDEO_SEGMENT_VIDEO_FRAMEWORK_CONCURRENT_QUEUE_H__
#include <list>

#include <boost/thread.hpp>
#include <boost/thread/thread_time.hpp>

template<typename Data>
class concurrent_queue {
private:
  // Underlying queue.
  std::list<Data> queue_;
  mutable boost::mutex mutex_;
  boost::condition_variable data_available_;
  boost::condition_variable data_consumed_;

public:
  void push(const Data& data) {
    boost::mutex::scoped_lock lock(mutex_);
    queue_.push_back(data);
    lock.unlock();
    data_available_.notify_one();
  }

  // Same as push, but locks caller until fewer than max_elems
  // have to be processed.
  void push(const Data& data, unsigned int max_elems) {
    boost::mutex::scoped_lock lock(mutex_);
    while (queue_.size() >= max_elems) {
      data_consumed_.wait(lock);
    }

    queue_.push_back(data);
    lock.unlock();
    data_available_.notify_one();
  }

  template <class Predicate>
  bool contains(const Predicate& pred) const {
    boost::mutex::scoped_lock lock(mutex_);
    typedef typename std::list<Data>::const_iterator const_iterator;
    for (const_iterator i = queue_.begin(); i != queue_.end(); ++i) {
      if (pred(*i))
        return true;
    }

    return false;
  }

  bool contains(const Data& data) const {
    return contains(std::bind2nd(std::equal_to<Data>(), data));
  }

  bool empty() const {
    boost::mutex::scoped_lock lock(mutex_);
    return queue_.empty();
  }

  int size() const {
    boost::mutex::scoped_lock lock(mutex_);
    return queue_.size();
  }

  int unsafe_size() const {
    return queue_.size();
  }

  bool try_pop(Data* popped_value) {
    boost::mutex::scoped_lock lock(mutex_);
    if(queue_.empty())
      return false;

    *popped_value = queue_.front();
    queue_.pop_front();

    // Notification for max_push operation.
    lock.unlock();
    data_consumed_.notify_one();
    return true;
  }

  void wait_and_pop(Data* popped_value) {
    boost::mutex::scoped_lock lock(mutex_);
    while(queue_.empty())
      data_available_.wait(lock);

    *popped_value = queue_.front();
    queue_.pop_front();

    // Notification for max_push operation.
    lock.unlock();
    data_consumed_.notify_one();
  }

  // Same as above but returns after wait_duration (in ms).
  // Returns true if item was successfully popped.
  bool timed_wait_and_pop(Data* popped_value,
                          int wait_duration = 1000) {
    const boost::system_time timeout = boost::get_system_time() +
        boost::posix_time::milliseconds(wait_duration);

    boost::mutex::scoped_lock lock(mutex_);
    while(queue_.empty()) {
      if(!data_available_.timed_wait(lock, timeout))
        return false;
    }

    *popped_value = queue_.front();
    queue_.pop_front();

    // Notification for max_push operation.
    lock.unlock();
    data_consumed_.notify_one();

    return true;
  }
};

#endif // VIDEO_SEGMENT_VIDEO_FRAMEWORK_CONCURRENT_QUEUE_H__
