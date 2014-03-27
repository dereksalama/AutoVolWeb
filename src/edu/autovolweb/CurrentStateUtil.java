package edu.autovolweb;

import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Collection;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import com.autovol.CurrentStateData;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

// Use this to convert to Instance so we don't need Weka lib on android
public class CurrentStateUtil {

	private static Collection<CurrentStateData> fromJson(String json) {
		Gson gson = new Gson();
		Type collectionType = new TypeToken<Collection<CurrentStateData>>(){}.getType();
		Collection<CurrentStateData> data = gson.fromJson(json, collectionType);
		return data;
	}
	
	private static void addInstance(CurrentStateData state, Instances dataset) {
		Instance i = toInstance(state, dataset);
		dataset.add(i);
	}
	
	public static Instance toInstance(CurrentStateData state) {
		return toInstance(state, createDataset());
	}
	
	private static Instance toInstance(CurrentStateData state, Instances dataset) {
		Instance i = new DenseInstance(CurrentStateData.NUM_ATTRS);
		i.setDataset(dataset);
		
		i.setValue(dataset.attribute("time"), state.getTime());
		i.setValue(dataset.attribute("lat"), state.getLat());
		i.setValue(dataset.attribute("lon"), state.getLon());
		i.setValue(dataset.attribute("loc_provider"), state.getLocProvider());
		i.setValue(dataset.attribute("light"), state.getLight());
		i.setValue(dataset.attribute("distance"), state.getLat());
		i.setValue(dataset.attribute("wifi_count"), state.getWifiCount());
		i.setValue(dataset.attribute("charging"), state.getCharging());
		i.setValue(dataset.attribute("activity_type"), state.getActivityType());
		i.setValue(dataset.attribute("activityConfidence"), state.getActivityConfidence());
		i.setValue(dataset.attribute("ringer"), state.getRinger());
		
		return i;
	}
	
	private static Instances createDataset() {
		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		
		attributes.add(new Attribute("time"));
		attributes.add(new Attribute("lat"));
		attributes.add(new Attribute("lat"));
		
		ArrayList<String> locProviderValues = new ArrayList<String>();
		locProviderValues.add("gps");
		locProviderValues.add("network");
		locProviderValues.add("fused");
		attributes.add(new Attribute("loc_provider", locProviderValues));
		
		attributes.add(new Attribute("light"));
		attributes.add(new Attribute("distance"));
		attributes.add(new Attribute("wifi_count"));
		
		ArrayList<String> chargingValues = new ArrayList<String>();
		chargingValues.add("true");
		chargingValues.add("false");
		attributes.add(new Attribute("charging", chargingValues));
		
		ArrayList<String> activityValues = new ArrayList<String>();
		chargingValues.add("activity_vehicle");
		chargingValues.add("activity_bike");
		chargingValues.add("activity_foot");
		chargingValues.add("activity_still");
		chargingValues.add("activity_unknown");
		chargingValues.add("activity_tilting");
		attributes.add(new Attribute("activity_type", activityValues));
		
		attributes.add(new Attribute("activity_confidence"));
		
		ArrayList<String> ringerValues = new ArrayList<String>();
		ringerValues.add("silent");
		ringerValues.add("vibrate");
		ringerValues.add("normal");
		Attribute ringerAttr = new Attribute("ringer", ringerValues);
		attributes.add(ringerAttr);
		
		Instances data = new Instances("Training", attributes, 0);
		data.setClass(ringerAttr);
		
		return data;
	}
	
	public static Instances convertCurrentStateData(String json) {
		Instances data = createDataset();
		
		Collection<CurrentStateData> states = fromJson(json);
		for (CurrentStateData state : states) {
			addInstance(state, data);
		}
		
		return data;
	}
}
