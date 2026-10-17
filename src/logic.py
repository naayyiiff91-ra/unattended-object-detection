# ownership/timer state machine
import time
import math


# Distance (in pixels) within which a bag is considered "attended" by a person
PROXIMITY_THRESHOLD = 150


def edge_distance(box_a, box_b):
    """
    Calculate the shortest distance between the edges of two bounding boxes.
    box_a, box_b: (x1, y1, x2, y2)
    Returns 0 if boxes overlap or touch.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    dx = max(bx1 - ax2, ax1 - bx2, 0)
    dy = max(by1 - ay2, ay1 - by2, 0)

    return math.sqrt(dx ** 2 + dy ** 2)


class BagState:
    """
    Tracks the ownership/attendance state of a single bag over time.
    States: 'attended', 'orphaned', 'unattended'
    """

    def __init__(self, bag_id):
        self.bag_id = bag_id
        self.state = "attended"
        self.orphaned_since = None  # timestamp when bag became orphaned
        self.last_seen = time.time()

    def update(self, is_near_person, unattended_threshold_seconds):
        """
        Update this bag's state based on whether a person is currently nearby.
        Returns the current state after update.
        """
        now = time.time()
        self.last_seen = now

        if is_near_person:
            # Person is close — bag is attended, reset any timer
            self.state = "attended"
            self.orphaned_since = None
        else:
            if self.state == "attended":
                # Just became orphaned — start the timer
                self.state = "orphaned"
                self.orphaned_since = now
            elif self.state == "orphaned":
                # Already orphaned — check if enough time has passed
                elapsed = now - self.orphaned_since
                if elapsed >= unattended_threshold_seconds:
                    self.state = "unattended"
            # if already 'unattended', stays unattended until a person returns

        return self.state

    def seconds_orphaned(self):
        """Returns how many seconds this bag has been orphaned/unattended, or 0 if attended."""
        if self.orphaned_since is None:
            return 0
        return time.time() - self.orphaned_since


class OwnershipTracker:
    """
    Manages BagState objects for all currently tracked bags,
    and determines attended/orphaned/unattended status each frame.
    """

    def __init__(self, proximity_threshold=PROXIMITY_THRESHOLD, unattended_threshold_seconds=30):
        self.bag_states = {}  # bag_id -> BagState
        self.proximity_threshold = proximity_threshold
        self.unattended_threshold_seconds = unattended_threshold_seconds

    def update(self, person_boxes, bag_boxes):
        """
        person_boxes: list of (track_id, (x1, y1, x2, y2))
        bag_boxes: list of (track_id, (x1, y1, x2, y2), class_name)

        Returns a dict: bag_id -> {
            'state': str,
            'seconds_orphaned': float,
            'class_name': str,
            'box': (x1, y1, x2, y2)
        }
        """
        results = {}
        current_bag_ids = set()

        for bag_id, bag_box, class_name in bag_boxes:
            current_bag_ids.add(bag_id)

            # Find if any person is within the proximity threshold of this bag
            is_near_person = False
            for person_id, person_box in person_boxes:
                dist = edge_distance(bag_box, person_box)
                if dist <= self.proximity_threshold:
                    is_near_person = True
                    break

            # Get or create state tracker for this bag
            if bag_id not in self.bag_states:
                self.bag_states[bag_id] = BagState(bag_id)

            state = self.bag_states[bag_id].update(
                is_near_person,
                self.unattended_threshold_seconds
            )

            results[bag_id] = {
                "state": state,
                "seconds_orphaned": self.bag_states[bag_id].seconds_orphaned(),
                "class_name": class_name,
                "box": bag_box
            }

        # Clean up state only for bags truly gone (not seen for > 3.0 seconds)
        # This prevents 1-frame tracker dropout from resetting the timer
        now = time.time()
        for stale_id in list(self.bag_states.keys()):
            if stale_id not in current_bag_ids:
                if now - self.bag_states[stale_id].last_seen > 3.0:
                    del self.bag_states[stale_id]

        return results