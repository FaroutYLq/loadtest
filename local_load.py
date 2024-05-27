import numpy as np
import sys
import gc
import cutax
import strax
from tqdm import tqdm

st = cutax.xenonnt_offline(output_folder='/dali/lgrandi/yuanlq/collected_outsource/finished_data',
                           _auto_append_rucio_local=False,
                           _rucio_local_path='/dali/lgrandi/rucio', include_rucio_local=True
                           )

print('Storage:')
print(st.storage)
print('--------------------')

loop_over = ['052738', '047621', '052745', '045067', '052747', '052750',
       '047631', '048660', '050714', '047643', '050718', '047647',
       '047648', '050719', '050720', '050721', '047653', '044582',
       '044071', '052776', '052777', '052778', '052779', '052780',
       '053293', '047661', '043055', '044584', '049201', '052782',
       '043571', '044589', '052785', '052786', '052787', '052788',
       '052789', '044598', '052790', '052284', '043581', '052285',
       '052791', '052288', '052792', '050747', '052291', '043588',
       '043589', '050751', '050758', '050760', '053321', '052299',
       '044107', '050765', '050769', '051794', '053331', '048211',
       '051797', '044118', '051799', '051800', '051801', '051802',
       '052313', '051804', '051805', '051807', '051808', '051810',
       '051811', '050786', '050790', '050791', '051822', '043119',
       '046719', '046720', '046721', '052868', '050823', '052363',
       '052365', '046734', '050831', '044687', '050833', '046735',
       '046738', '051860', '051861', '050836', '050837', '049304',
       '051864', '049306', '051865', '044184', '052377', '052378',
       '044189', '050839', '050840', '052897', '052898', '049316',
       '051877', '051878', '051879', '049320', '052902', '052903',
       '052908', '046765', '052910', '046768', '047282', '052915',
       '043700', '052920', '052922', '049864', '049871', '049872',
       '052949', '045276', '048351', '044256', '052959', '052969',
       '052970', '048365', '048366', '052973', '045296', '052974',
       '047348', '049403', '048382', '053503', '053504', '053505',
       '048385', '048387', '050945', '047365', '048389', '046739',
       '043822', '048430', '046900', '048439', '043839', '052544',
       '052545', '053055', '053060', '048456', '052553', '052554',
       '052558', '048463', '053072', '053075', '046932', '053078',
       '052567', '052568', '043865', '053082', '048476', '050748',
       '046943', '046944', '048484', '048486', '048487', '050025',
       '050026', '050027', '050028', '047473', '047474', '050036',
       '047477', '052084', '050039', '050040', '052086', '053116',
       '050046', '050047', '053120', '052100', '052613', '053126',
       '047495', '052104', '047497', '052617', '053133', '047506',
       '052794', '050070', '048535', '050074', '047515', '052796',
       '048546', '046499', '050085', '050086', '050092', '052149',
       '048567', '047548', '047551', '052159', '047556', '047565',
       '052176', '047573', '050772', '050655', '053215', '050658',
       '050662', '050663', '052715', '052716', '052717', '052719',
       '052721', '052724', '052726', '052728', '052729', '044026']

must_have_peaks = ["peaklets", "lone_hits"]
must_have_event = ["peaklets", "lone_hits", "merged_s2s", "peaklet_classification", "peak_basics", "peak_positions_mlp", "peak_positions_cnn", "peak_positions_gcn", "event_info", "cuts_basic"]

peaks_not_done = []
event_not_done = []
corrupted = []
can_deliver_both = []
can_deliver_peaklets = []

for r in tqdm(loop_over):
	print('Loading run %s' % r)

	must_have_peaks_satisfied = True
	must_have_event_satisfied = True
	# These will be rerun on OSG or dali to reprocess from raw_records
	for d in must_have_peaks:
		if not st.is_stored(r, d):
			print('Run %s does not have %s' % (r, d))
			must_have_peaks_satisfied = False
			peaks_not_done.append(r)
			break

	if must_have_peaks_satisfied:
		can_deliver_peaklets.append(r)
	
	# These have peaklets+lone_hits but not finished events on OSG
	# We want to process them on dali
	if must_have_peaks_satisfied:
		for d in must_have_event:
			if not st.is_stored(r, d):
				print('Run %s does not have %s' % (r, d))
				must_have_event_satisfied = False
				event_not_done.append(r)
				break

	# Check if the run is corrupted
	is_corrupted = False
	if must_have_event_satisfied:
		for tu in [("peaks", "peak_positions", "peak_basics"), ("event_info", "cuts_basic")]:
			try:
				_loaded = st.get_array(r, tu, keep_columns=("time"))
				del _loaded
				gc.collect()
			except Exception as e:
				print('Run %s corrupted: %s' % (r, e))
				corrupted.append(r)
				is_corrupted = True
				break
	if not is_corrupted:
		can_deliver_both.append(r)

print("Can deliver peaklets: %d" % len(can_deliver_peaklets))
print(can_deliver_peaklets)

print("Can deliver both: %d" % len(can_deliver_both))
print(can_deliver_both)