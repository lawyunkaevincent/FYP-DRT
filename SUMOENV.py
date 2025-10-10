import os, sys, math, time, collections, random
import numpy as np
import traci
import queue

class SumoTaxiEnv:
    def __init__(self, cfg_path, step_length=1.0, use_gui=False):
        self.cfg_path = cfg_path
        self.step_length = step_length 
        self.action_space = [1, 2, 3, 4, 5]
        self.use_gui = use_gui
        self.done = False
        self.state = None
        self.started = False
        # this is a queue that store simulation time for computing reward later(after two decision points)
        self.metrics_queue = queue.Queue(maxsize=3)
        self.start_metrics = None
        self.first_update = False
        self._last_decision_context = None
        self.time_diff_list = []
        self.mean_wait_dec_list = []
        self.oldest_wait_dec_list = []
        # key is taxis, value is current onboarding person id
        self.LUT_dict = {}
        # for reward computation
        self._ema = {"td":0.0,"mw":0.0,"ow":0.0}
        self._v = {"td":1.0,"mw":1.0,"ow":1.0}   # variance estimates
        self._ema_inited = False


    # ---------- SUMO glue ----------
    def _start_sumo(self):
        sumo_bin = "sumo-gui" if self.use_gui else "sumo"
        traci.start([sumo_bin, "-c", self.cfg_path, "--step-length", str(self.step_length)])
        self.started = True

    def _close_sumo(self):
        if self.started:
            traci.close()
            self.started = False
        if hasattr(self, "log_f") and self.log_f:
            self.log_f.close()

    def _compute_unassigned_req(self, requests):
        unassigned_req = []
        for req in requests:
            if req.state == 2:
                unassigned_req.append(req)
        return unassigned_req
        
    def _step_until_decision(self):
        """
        Advance SUMO until a decision epoch: ≥1 idle taxi OR (optionally) new requests.
        Return (idle_taxis, pending_requests).
        If there are 1 idle taxi and 1 requests, we can directly dispatch without waiting (no need computation)
        """
        # Loop until condition
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            # see if there is any idle taxi
            idle_taxis = traci.vehicle.getTaxiFleet(0)  # all idle taxis
            requests = traci.person.getTaxiReservations(0)  # 0 for all the reservations that are active
            unassigned_req = self._compute_unassigned_req(requests)
            # print(f"\nAll Request here: {requests}")
            # print(f"Unassigned Request here: {unassigned_req}\n")
            if len(idle_taxis) >= 1 and len(unassigned_req) >= 1:
                return idle_taxis, unassigned_req, False
        return [], [], True  # no more expected vehicles
        

    # ---------- State / Actions / Reward ----------
    def _bin(self, x, edges):           # helper: map numeric to bin index
        # edges like [30,60,120,9999]
        for i, e in enumerate(edges):
            if x <= e: return i
        return len(edges)
    
    def _get_oldest_request(self, requests):
        max_wait = -1
        max_req = -1
        for req in requests:
            wait_time = traci.person.getWaitingTime(req.persons[0])
            max_wait = max(max_wait, wait_time)
            max_req = req if wait_time == max_wait else max_req
        return max_wait, max_req
    
    def _get_person_destination_info(self, person_id):
        # the request here is special it is all the reqeust
        requests = traci.person.getTaxiReservations()
        for req in requests:
            req_person_id = req.persons[0]
            # print(f"req_person_id: {req_person_id} && person_id: {person_id}")
            if req_person_id == person_id:
                return req.toEdge, req.arrivalPos
    
    def _compute_separation(self):
        all_taxis = traci.vehicle.getTaxiFleet(-1)
        taxi0 = all_taxis[0]
        taxi1 = all_taxis[1]
        idle_taxis = traci.vehicle.getTaxiFleet(0)
        if taxi0 in idle_taxis and taxi1 in idle_taxis:
            x0, y0 = traci.vehicle.getPosition(taxi0)
            x1, y1 = traci.vehicle.getPosition(taxi1)
            return traci.simulation.getDistance2D(x0, y0, x1, y1, isDriving=True)
        idle_taxi = traci.vehicle.getTaxiFleet(0)[0]
        # print(f"IdleTaxi: {idle_taxi}")
        # print(f"OccupiedTaxi: {traci.vehicle.getTaxiFleet(2)}")
        pickedup_taxi = traci.vehicle.getTaxiFleet(1)
        occupied_taxi = traci.vehicle.getTaxiFleet(2)
        if occupied_taxi:
            occupied_taxi_id = occupied_taxi[0]
        else:
            occupied_taxi_id = pickedup_taxi[0]

        occupied_taxi_passenger_id = self.LUT_dict[occupied_taxi_id]
        # print(f"OccupiedTaxiId: {occupied_taxi_id}")
        # print(f"OccupiedTaxiPassengerId: {occupied_taxi_passenger_id}")
        
        ret = self._get_person_destination_info(occupied_taxi_passenger_id)
        if ret:
            taxi_dest_edge, taxi_dest_pos = ret
        else:
            taxi_dest_edge = traci.vehicle.getRoadID(occupied_taxi_passenger_id)
            taxi_dest_pos = traci.vehicle.getLanePosition(occupied_taxi_passenger_id)
        idle_taxi_edge = traci.vehicle.getRoadID(idle_taxi)
        idle_taxi_pos = traci.vehicle.getLanePosition(idle_taxi)
        return traci.simulation.getDistanceRoad(idle_taxi_edge, idle_taxi_pos, taxi_dest_edge, taxi_dest_pos)
        

    def _build_state(self, idle_taxis, requests):
        """
        Build a small tuple of binned features.
        """
        idle_count = len(idle_taxis)
        pending_req_count = len(requests)
        max_wait, _ = self._get_oldest_request(requests)
        oldest_wait_bin = self._bin(max_wait, [100,400,900,1500])
        taxi_sep_bin = self._bin(self._compute_separation(),[300, 400, 550])
        s = (idle_count, pending_req_count, oldest_wait_bin, taxi_sep_bin)
        return s
    
    def _compute_nearest_request(self, idle_taxi, requests):
        min_eta = float('inf')
        # print(f"Requests: {requests}")
        nearest_req = requests[0]
        vType = traci.vehicle.getTypeID(idle_taxi)
        for req in requests:
            route = traci.simulation.findRoute(traci.vehicle.getRoadID(idle_taxi), req.fromEdge, vType, routingMode=1)
            eta = route.travelTime
            if eta < min_eta:
                min_eta = eta
                nearest_req = req
        return nearest_req
    
    def _compute_furthest_request_with_taxi2(self, idle_taxi, requests):
        try:
            vType = traci.vehicle.getTypeID(idle_taxi)
            occupied_taxi = traci.vehicle.getTaxiFleet(2)
            pickedup_taxi = traci.vehicle.getTaxiFleet(1)
            # print(f"Occupied_taxi: {occupied_taxi}")
            # print(f"Picked Up taxi: {pickedup_taxi}")
            if occupied_taxi:
                occupied_taxi = occupied_taxi[0]
            # use the currently pick passenger taxi
            else:
                occupied_taxi = pickedup_taxi[0]
            # print(f"Occupied taxi: {occupied_taxi}")
            # print(f"Passenger id: {self.LUT_dict[occupied_taxi]}")
            occupied_taxi_passenger_id = self.LUT_dict[occupied_taxi]
            taxi_dest_edge, _ = self._get_person_destination_info(occupied_taxi_passenger_id)

        except IndexError:
            idle_taxis = traci.vehicle.getTaxiFleet(0)
            # print(f"Choose From Idle taxi:{idle_taxis}")
            the_other_taxi = idle_taxis[0] if idle_taxi != idle_taxis[0] else idle_taxis[1]
            taxi_dest_edge = traci.vehicle.getRoadID(the_other_taxi)

        finally:
            max_eta = -1
            furthest_req = requests[0]
            for req in requests:
                route = traci.simulation.findRoute(taxi_dest_edge, req.fromEdge, vType, routingMode=1)
                eta = route.travelTime
                if eta > max_eta:
                    max_eta = eta
                    furthest_req = req


        return furthest_req
    
    def _compute_median_request(self, idle_taxi, requests):
        wait_times = []
        distances = []
        vType = traci.vehicle.getTypeID(idle_taxi)
        idle_taxi_edge = traci.vehicle.getRoadID(idle_taxi)
        for req in requests:
            wait_time = traci.person.getWaitingTime(req.persons[0])
            wait_times.append(wait_time)
            route = traci.simulation.findRoute(idle_taxi_edge, req.fromEdge, vType, routingMode=1)
            distance = route.travelTime
            distances.append(distance)
        weighted_wait_times_distance = [(0.7 * wait_times[i] + 0.3 * distances[i], requests[i]) for i in range(len(requests))]
        weighted_wait_times_distance.sort(key=lambda x: x[0])
        # print(f"Weighted wait times and distances: {weighted_wait_times_distance}")
        return weighted_wait_times_distance[len(weighted_wait_times_distance) // 2][1]

    def _take_action(self, action, requests, idle_taxis):
        # handle one taxis at a time
        # action 1 -> dispatch oldest
        idle_taxi = idle_taxis[0]
        if action == 1:
            _, r_oldest = self._get_oldest_request(requests)
            if r_oldest:
                traci.vehicle.dispatchTaxi(idle_taxi, r_oldest.id)
                self.LUT_dict[idle_taxi] = r_oldest.id
        
        # action 2 -> dispatch nearest (by ETA)
        elif action == 2:
            r_nearest = self._compute_nearest_request(idle_taxi, requests)
            if r_nearest:
                traci.vehicle.dispatchTaxi(idle_taxi, r_nearest.id)
                self.LUT_dict[idle_taxi] = r_nearest.id
                
        # action 3 -> pick up customer who is furthest away from the other taxi
        elif action == 3:
            r_furthest = self._compute_furthest_request_with_taxi2(idle_taxi, requests)
            if r_furthest:
                traci.vehicle.dispatchTaxi(idle_taxi, r_furthest.id)
                self.LUT_dict[idle_taxi] = r_furthest.id
            
        # action 4 -> pick up a customer that has the median wait time as well as median distance from the idle taxi (weighted)
        elif action == 4:
            r_middle = self._compute_median_request(idle_taxi, requests)
            if r_middle:
                traci.vehicle.dispatchTaxi(idle_taxi, r_middle.id)
                self.LUT_dict[idle_taxi] = r_middle.id
                
        # action 5 -> do nothing
        elif action == 5:
            pass
        
     
    def _z(self, key, x, beta=0.99):
        if not self._ema_inited:
            self._ema[key] = x
            self._v[key] = (x - self._ema[key])**2 + 1e-4
            return 0.0
        mu = self._ema[key]
        var = self._v[key]
        z = (x - mu) / (np.sqrt(var) + 1e-6)
        # update AFTER computing z so current sample doesn’t normalize itself
        self._ema[key] = beta*mu + (1-beta)*x
        self._v[key] = beta*var + (1-beta)*(x - self._ema[key])**2
        return float(np.clip(z, -3.0, 3.0))

    def _compute_reward(self, prev_metrics, curr_metrics):
        td = curr_metrics["time"] - prev_metrics["time"]
        d_mean = prev_metrics["mean_wait"] - curr_metrics["mean_wait"]
        d_old = prev_metrics["oldest_wait"] - curr_metrics["oldest_wait"]

        if not self._ema_inited:
            # warmup: first call initializes EMA, return a small neutral reward
            self._ema_inited = True
            _ = self._z("td", td); _ = self._z("mw", d_mean); _ = self._z("ow", d_old)
            return 0.0

        r = -0.1*self._z("td", td) + 0.5*self._z("mw", d_mean) + 0.3*self._z("ow", d_old)
        return float(np.clip(r, -1.0, 1.0))


    def _collect_metrics(self):
        """
        Return dict with:
          - total_wait_sum (sum over all pending requests)
          - pickups_done (cumulative)
        """
        # time_diff with previous state，average waiting time, maximum waiting time, number of pickups
        cur_time = traci.simulation.getTime()
        per_ids = traci.person.getIDList()
        if per_ids:
            wait_times = [traci.person.getWaitingTime(pid) for pid in per_ids]
            oldest_wait = max(wait_times)
            mean_wait = sum(wait_times) / len(wait_times)
        else:
            oldest_wait = 0 
            mean_wait = 0
        return {
            "time": cur_time,
            "oldest_wait": oldest_wait,
            "mean_wait": mean_wait,}

    # ---------- Gym-like API ----------
    def reset(self):
        if self.started:
            self._close_sumo()
        self._start_sumo()
        idle, reqs, done = self._step_until_decision()
        self._last_decision_context = (idle, reqs, done)
        self.LUT_dict = {}
        self.state = self._build_state(idle, reqs)
        self.metrics_queue = queue.Queue(maxsize=3)
        self.start_metrics = None
        self.first_update = False
        self.done = False
        self.time_diff_list = []
        self.mean_wait_dec_list = []
        self.oldest_wait_dec_list = []

        return self.state

    def step(self, action):
        idle, reqs, _ = self._last_decision_context
        # print(f"Idle: {idle}, Reqs: {reqs}")
        self._take_action(action, reqs, idle)
        # reach next decision epoch
        idle, reqs, done = self._step_until_decision()
        next_state = self._build_state(idle, reqs)
        curr_metrics = self._collect_metrics()
        self.metrics_queue.put(curr_metrics)
        if not self.start_metrics:
            self.start_metrics = curr_metrics
        if self.metrics_queue.full():
            prev_metrics = self.metrics_queue.get()
            reward = self._compute_reward(prev_metrics, curr_metrics)
        elif not self.first_update:
            self.first_update = True
            reward = self._compute_reward(self.start_metrics, curr_metrics)
        else:
            reward = 0.0
        self._last_decision_context = (idle, reqs, done)
        return next_state, reward, done

    def close(self):
        self._close_sumo()
