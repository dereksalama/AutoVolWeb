package edu.autovolweb;

public class CurrentStateData {
	

	public static final int NUM_ATTRS = 15;

	private double day;
	private double time;
	private double lat;
	private double lon; // "long"
	private String locProvider;
	private double light;
	private double distance;
	private double wifiCount;
	private String charging;
	private String activityType;
	private double activityConfidence;
	private double audioMag;
	private String screenOn;
	private double screenLastOn;
	private String ringer;
	
	public CurrentStateData() {} 
	
	// Deep copy
	public CurrentStateData(CurrentStateData data) {
		day = data.getDay();
		time = data.getTime();
		lat = data.getLat();
		lon = data.getLon();
		locProvider = data.getLocProvider();
		light = data.getLight();
		distance = data.getDistance();
		wifiCount = data.getWifiCount();
		charging = data.getCharging();
		activityType = data.getActivityType();
		activityConfidence = data.getActivityConfidence();
		audioMag = data.getAudioMag();
		screenOn = data.getScreenOn();
		screenLastOn = data.getScreenLastOn();
		ringer = data.getRinger();
	}
	
	public double getTime() {
		return time;
	}

	public void setTime(double time) {
		this.time = time;
	}

	public double getLat() {
		return lat;
	}

	public void setLat(double lat) {
		this.lat = lat;
	}

	public double getLon() {
		return lon;
	}

	public void setLon(double lon) {
		this.lon = lon;
	}

	public String getLocProvider() {
		return locProvider;
	}

	public void setLocProvider(String locProvider) {
		this.locProvider = locProvider;
	}

	public double getLight() {
		return light;
	}

	public void setLight(double light) {
		this.light = light;
	}

	public double getDistance() {
		return distance;
	}

	public void setDistance(double distance) {
		this.distance = distance;
	}

	public double getWifiCount() {
		return wifiCount;
	}

	public void setWifiCount(double wifiCount) {
		this.wifiCount = wifiCount;
	}

	public String getCharging() {
		return charging;
	}

	public void setCharging(String charging) {
		this.charging = charging;
	}

	public String getActivityType() {
		return activityType;
	}

	public void setActivityType(String activityType) {
		this.activityType = activityType;
	}

	public double getActivityConfidence() {
		return activityConfidence;
	}

	public void setActivityConfidence(double activityConfidence) {
		this.activityConfidence = activityConfidence;
	}

	public String getRinger() {
		return ringer;
	}

	public void setRinger(String ringer) {
		this.ringer = ringer;
	}

	public double getAudioMag() {
		return audioMag;
	}

	public void setAudioMag(double audioMag) {
		this.audioMag = audioMag;
	}

	public String getScreenOn() {
		return screenOn;
	}

	public void setScreenOn(String screenOn) {
		this.screenOn = screenOn;
	}

	public double getDay() {
		return day;
	}

	public void setDay(double day) {
		this.day = day;
	}

	public double getScreenLastOn() {
		return screenLastOn;
	}

	public void setScreenLastOn(double screenLastOn) {
		this.screenLastOn = screenLastOn;
	}
}
