# control.py
# rule-based controller and energy calc
def compute_energy(devices):
    # devices: dict like {"lights":1,"fan":1,"ac":1,"ac_power_kw":0.8}
    # return energy per hour in kWh (approx)
    energies = {}
    energies['lights_kwh'] = 0.2 * devices.get('lights',0) # 200W when ON
    energies['fan_kwh'] = 0.075 * devices.get('fan',0)     # 75W single fan
    energies['ac_kwh'] = devices.get('ac_power_kw', 0) if devices.get('ac',0) else 0.0
    total = sum(energies.values())
    energies['total_kwh'] = round(total,4)
    return energies

def rule_based_control(state, predicted_occupancy):
    """
    state: dict with keys: occupancy, motion, temp, co2, solar_kw, battery_soc
    predicted_occupancy: int (future occupancy)
    returns: device actions dict
    """
    devices = {"lights":0,"fan":0,"ac":0,"ac_power_kw":0.0}
    occ = predicted_occupancy if predicted_occupancy is not None else state.get('occupancy',0)
    # lights: ON if predicted occupancy > 0
    devices['lights'] = 1 if occ >= 1 else 0
    # fans: ON if occupancy >= 3
    devices['fan'] = 1 if occ >= 3 else 0
    # AC: use if occ >= 10 or temp > 26
    if occ >= 10 or state.get('temp',25) > 26:
        devices['ac'] = 1
        # AC power adjust by occupancy: base 1.2 kW, scale a bit
        devices['ac_power_kw'] = round(1.2 + 0.005*occ,3)
    else:
        devices['ac'] = 0
        devices['ac_power_kw'] = 0.0
    # energy source decision simple:
    # If solar_kw > total_kwh and battery_soc>0.2 prefer solar
    energy = compute_energy(devices)
    use_solar = False
    if state.get('solar_kw',0) >= energy['total_kwh'] or state.get('battery_soc',0) > 0.2:
        use_solar = True
    return {"devices":devices, "energy": energy, "use_solar": use_solar}
