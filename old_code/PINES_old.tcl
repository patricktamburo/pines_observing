#PINES.tcl contains tcltk procedures for conducting the PINES observing program on the Perkins
#The procedures read text files located in the current data directory. 

# proc PINES_set_date {set_date} {
#     global pines_date
#     set pines_date $set_date
#     puts [concat "PINES date (YYYYMMDD in UT) set to" $set_date]
# }

#Replaces PINES_set_date
proc PINES_get_date {} {

    #gets date in UT YYYYMMDD for writing reading files in the data directory
    
    #Add 10 hours to system time to ensure UT at sunset and sunrise
    set systemTime [expr [clock seconds] + 12 * 3600]
    set yr [clock format $systemTime -format %Y]
    set mn [clock format $systemTime -format %m]
    set day [clock format $systemTime -format %d]
    return $yr$mn$day
    
}

proc PINES_watchdog_get {start_file_time} {

    set pines_date [PINES_get_date]
    
    puts "Waiting until PINES_watchdog calculates shift/fwhm (10 sec timeout)..."
    set start_wait_time [ clock seconds ]
    set current_wait_time [ clock seconds ]
    set current_file_time [file mtime /mimir/data/obs72/$pines_date/image_shift.txt]
    while { $current_wait_time < $start_wait_time + 10 && $current_file_time == $start_file_time } {
	#Pause for 0.1 second within forloop to slow it down
	after 100
	set current_wait_time [ clock seconds ]
	set current_file_time [file mtime /mimir/data/obs72/$pines_date/image_shift.txt]
    }

    #Wait 0.5 seconds for file to be completely written
    after 500
    
    puts "Reading in text written by python program..."
    set fid [open /mimir/data/obs72/$pines_date/image_shift.txt r]
    set text [read $fid]
    close $fid
    set list [split $text " "]
    
    set d_ra [expr [lindex $list 0]]
    set d_dec [expr [lindex $list 1]]
    set name [lindex $list 2]
    set fwhm [lindex $list 3]
    
    return [list $d_ra $d_dec $name $fwhm]

}

proc PINES_focus {exptime current_focus} {

    puts "STARTING PINES_focus..."

    set pines_date [PINES_get_date]

    #create list with focus positions
    set rel_focus_list { -200 -150 -100 -50 0 50 100 150 200}

    #create index
    set index 1

    #move focus inside start point, so we are always moving it out
    focus_go [expr $current_focus + [lindex $rel_focus_list 0] - 50]
    
    #iterate through focus positions
    foreach rel_focus $rel_focus_list {
	focus_go [expr $current_focus + $rel_focus]
	    
	#put zeros into old file
	exec echo 0.0 0.0 dummy 0.0 > /mimir/data/obs72/$pines_date/image_shift.txt

	#save time the file was modified
	set start_file_time [file mtime /mimir/data/obs72/$pines_date/image_shift.txt]

	puts "Taking a test exposure..."	    
	test etime=$exptime nexp=1 {comment1=} {comment2=}


	#read output from PINES_watchdog
	set data [PINES_watchdog_get $start_file_time]
	set d_ra [expr [lindex $data 0]]
	set d_dec [expr [lindex $data 1]]
	set target_name [lindex $data 2]
	set fwhm [lindex $data 3]

	lappend fwhm_list $fwhm

	puts [concat "Focus=" [expr $current_focus + $rel_focus] ", FWHM=" $fwhm]

	#increment index
	incr index
    }
    
    #print out focus locations and fwhm
    set index 0
    puts "Focus FWHM:"
    foreach rel_focus $rel_focus_list {
	puts [concat [expr $current_focus + $rel_focus] [lindex $fwhm_list $index]]
	#increment index
	incr index
    }
    
    #find minimum fwhm
    set fwhm_sorted [lsort $fwhm_list]

    set best_index [lsearch $fwhm_list [lindex $fwhm_sorted 0]]
    set best_rel_focus [lindex $rel_focus_list $best_index]

    set index 1
    while {[lindex $fwhm_list $best_index] == -1} {
	set best_index [lsearch $fwhm_list [lindex $fwhm_sorted $index]]
	set best_rel_focus [lindex $rel_focus_list $best_index]
	incr index
	if {$index == 4} {break}
    }


    #move to best focus
    puts [concat "Moving to best focus of" [expr $current_focus + $best_rel_focus] "..."]
    focus_go [expr $current_focus + $best_rel_focus - 50]
    focus_go [expr $current_focus + $best_rel_focus]

    puts "PINES_focus COMPLETE."

    #Three quarks for muster mark.
    for {set i 0} {$i < 3} {incr i} {
	puts \a\a
	after 300
    }


}


proc PINES_peakup {exptime} {

    set pines_date [PINES_get_date]

    puts "STARTING PINES_peakup..."

    set prop 0.9

    #set maximum move for peak up in arcseconds
    set limit 300

    #put zeros into old file
    exec echo 0.0 0.0 dummy 0.0 > /mimir/data/obs72/$pines_date/image_shift.txt

    #save time the file was modified
    set start_file_time [file mtime /mimir/data/obs72/$pines_date/image_shift.txt]

    puts "Taking a test exposure..."	    
    test etime=$exptime nexp=1 {comment1=} {comment2=}

    #read output from PINES_watchdog
    set data [PINES_watchdog_get $start_file_time]
    set d_ra [expr [lindex $data 0]]
    set d_dec [expr [lindex $data 1]]
    set target_name [lindex $data 2]
    set fwhm [lindex $data 3]

    puts [concat "Target name is" $target_name "."]

    #Move if the suggested motion is good

    if {[expr abs($d_ra)] == 0 && [expr abs($d_dec)] == 0
    } then {puts "Suggested RA and DEC moves equal to 0. PINES_watchdog likely failed. Move not executed."
    } elseif { [expr abs($d_ra)] < $limit && [expr abs($d_dec)] < $limit 
	   } then {concat "Moving telescope" [expr $d_ra * $prop] "arcsec in RA and" [expr $d_ra * $prop] "in DEC..."
	rmove d_ra=[expr $d_ra * $prop] d_dec=[expr $d_dec * $prop]	
    } else {puts [concat "Suggested RA or DEC move greater than" $limit "arcsec limit. Move not executed."]}

    #Now run a second time    
    #save time the file was last modified
    set start_file_time [file mtime /mimir/data/obs72/$pines_date/image_shift.txt]

    puts "Taking a test exposure..."	    
    test etime=$exptime nexp=1 title=$target_name {comment1=} {comment2=}

    #read output from PINES_watchdog
    set data [PINES_watchdog_get $start_file_time]
    set d_ra [expr [lindex $data 0]]
    set d_dec [expr [lindex $data 1]]
    set target_name [lindex $data 2]
    set fwhm [lindex $data 3]

    puts [concat "Target name is" $target_name "."]

    if {[expr abs($d_ra)] == 0 && [expr abs($d_dec)] == 0
    } then {puts "Suggested RA and DEC moves equal to 0. PINES_watchdog likely failed. Move not executed."
    } elseif { [expr abs($d_ra)] < $limit && [expr abs($d_dec)] < $limit 
	   } then {concat "Moving telescope" [expr $d_ra * $prop] "arcsec in RA and" [expr $d_ra * $prop] "in DEC..."
	rmove d_ra=[expr $d_ra * $prop] d_dec=[expr $d_dec * $prop]	
    } else {puts [concat "Suggested RA or DEC move greater than" $limit "arcsec limit. Move not executed."]}
    

    #Now run a third time to test for success
    #save time the file was last modified
    set start_file_time [file mtime /mimir/data/obs72/$pines_date/image_shift.txt]

    puts "Taking a test exposure..."	    
    test etime=$exptime nexp=1 title=$target_name {comment1=} {comment2=}

    #read output from PINES_watchdog
    set data [PINES_watchdog_get $start_file_time]
    set d_ra [expr [lindex $data 0]]
    set d_dec [expr [lindex $data 1]]
    set target_name [lindex $data 2]
    set fwhm [lindex $data 3]

    set separation [expr sqrt(pow($d_ra,2) + pow($d_dec,2))]
    
    if {$separation == 0
    } then { puts "PINES_peakup unsuccessful.  PINES_watchdog returning 0 0."
	set success -1
    } elseif { $separation < 10.0
	   } then { puts "PINES_peakup successful, target within 10 arcseconds of master position."
	set success 1
    } else {puts [concat "PINES_peakup unsuccessful, target" $separation "arcseconds away from master position."]
	set success -1}
    
    puts "PINES_peakup COMPLETE."

    #Three quarks for muster mark.
    for {set i 0} {$i < 3} {incr i} {
	puts \a\a
	after 300
    }

    return $success
    
}

proc PINES_guide {exptime total_time} {

    puts "STARTING PINES_guide..."

    set pines_date [PINES_get_date]
    set prop 0.9

    #set limits for guiding corrections in arcseconds
    set lower_limit 1.5
    set upper_limit 60.0
    
    #get start time
    set start_time [ clock seconds ]

    puts [concat "Taking" $exptime "second exposures for" $total_time "seconds..."]

    #set index to 1
    set index 1

    #get current time for while loop
    set current_time [ clock seconds ]

    #start while loop
    while {$current_time < $start_time + $total_time - $exptime} {

        #put zeros into old file (in case watchdog hanges, won't slew)
        exec echo 0.0 0.0 dummy 0.0 > /mimir/data/obs72/$pines_date/image_shift.txt
        #save time the file was modified
        set start_file_time [file mtime /mimir/data/obs72/$pines_date/image_shift.txt]

	#take exposure
	puts [concat "Taking exposure for" $exptime "seconds"]
        go etime=$exptime nexp=1 {comment1=} {comment2=}
  
	#read output from PINES_watchdog
	set data [PINES_watchdog_get $start_file_time]
	set d_ra [expr [lindex $data 0]]
	set d_dec [expr [lindex $data 1]]
	set target_name [lindex $data 2]
	set fwhm [lindex $data 3]

	#append these values to lists
	lappend d_ra_list $d_ra
	lappend d_dec_list $d_dec
	
	puts [concat $d_ra $d_dec]

        # every 3rd exposure, see if target has drifted
        if {$index % 3 == 0} {

	    #take mean of last three ra and dec suggestions
	    # set d_ra_1 [lindex $d_ra_list end]
	    # set d_ra_2 [lindex $d_ra_list end-1] 
	    # set d_ra_3 [lindex $d_ra_list end-2]
	    # set d_ra_mean [expr ($d_ra_1 + $d_ra_2 + $d_ra_3) / 3]

	    # set d_dec_1 [lindex $d_dec_list end]
	    # set d_dec_2 [lindex $d_dec_list end-1] 
	    # set d_dec_3 [lindex $d_dec_list end-2]
	    # set d_dec_mean [expr ($d_dec_1 + $d_dec_2 + $d_dec_3) / 3]

	    #take median of last three ra and dec suggestions
	    set d_ra_median [lindex [lsort -real [lrange $d_ra_list end-2 end]] 1]
	    set d_dec_median [lindex [lsort -real [lrange $d_dec_list end-2 end]] 1]

            # if {[expr abs($d_ra_median)] > $lower_limit || [expr abs($d_dec_median)] > $lower_limit
            # } then {puts [concat "Moving telescope median of last three suggestions: " [expr $d_ra_median * $prop] "arcsec in RA and" [expr $d_dec_median * $prop] "arcsec in Dec..."]
	    # 	            rmove d_ra=[expr $d_ra_median * $prop] d_dec=[expr $d_dec_median * $prop]
            # } else {puts [concat "Median of last three suggestions less than" $lower_limit "arcsec lower_limit. Move not executed..."]}

	    if {[expr abs($d_ra_median)] > $lower_limit && [expr abs($d_ra_median)] < $upper_limit
	    } then {puts [concat "Moving telescope median of last three suggestions: " [expr $d_ra_median * $prop] "arcsec in RA..."]
		rmove d_ra=[expr $d_ra_median * $prop]
	    } else {puts [concat "Median of last three RA suggestions" [expr $d_ra_median] "arcsec outside guide limits."]}

	    if {[expr abs($d_dec_median)] > $lower_limit && [expr abs($d_dec_median)] < $upper_limit
	    } then {puts [concat "Moving telescope median of last three suggestions: " [expr $d_dec_median * $prop] "arcsec in DEC..."]
		rmove d_dec=[expr $d_dec_median * $prop]
	    } else {puts [concat "Median of last three DEC suggestions" [expr $d_dec_median] "arcsec outside guide limits."]}


        }

        #get current time
	set current_time [ clock seconds ]

	#print out time left
        set time_left [expr $start_time + $total_time - $current_time ]
        puts [concat "Time left:" $time_left "seconds..."]

	#increment index
	incr index
    }

    puts "PINES_guide COMPLETE."

    #Three quarks for muster mark.
    for {set i 0} {$i < 3} {incr i} {
	puts \a\a
	after 300
    }

}




#program to calculate LST using computer system time and assuming mountain standard (GMT-7)
proc PINES_lst {} {

    #Time zone (hours east of Greenwich, -6 for MST)
    set tz {-7}
    set lng -111.53592

    set systemTime [clock seconds]    
    set yr [scan [clock format $systemTime -format %Y] %d]
    set mn [scan [clock format $systemTime -format %m] %d]
    set day [scan [clock format $systemTime -format %d] %d]
    set hr [scan [clock format $systemTime -format %H] %d]
    set min [scan [clock format $systemTime -format %M] %d]
    set sec [scan [clock format $systemTime -format %S] %d]

    set hr [expr {$hr} + {$min} / 60.0 + {$sec} / 3600.0 - $tz]
    #set hr [expr [expr [scan $hr %d]] + [expr [scan $min %d]] / 60.0 + [expr [scan $sec %d]] / 3600.0 - $tz]  

    if {$mn == 1 || $mn == 2
    } then {set L -1
    } else {set L 0}

    set julian [expr $day - 32075 + 1461*($yr+4800+$L)/4 + 367*($mn - 2 - $L*12)/12 - 3*(($yr + 4900 + $L)/100)/4]
    #set julian [expr [scan $day %d] - 32075 + 1461*($yr+4800+$L)/4 + 367*([scan $mn %d] - 2 - $L*12)/12 - 3*(($yr + 4900 + $L)/100)/4]

    set jd [expr $julian + ($hr / 24.0) - 0.5]
    #puts [concat "JD=" $jd]
    
    set c {280.46061837 360.98564736629 0.000387933 38710000.0}

    set jd2000 2451545.0

    set t0 [expr $jd - $jd2000]

    set t [expr $t0/36525]

    set theta [expr [lindex $c 0] + ([lindex $c 1] * $t0) + pow($t,2)*([lindex $c 2] - $t / [lindex $c 3])]

    set lst [expr ($theta + $lng) / 15.0]

    set lst [expr ([expr int($lst * 1000)] % 24000)/1000.0]
    
    puts [concat "LST=" $lst]

    return $lst
}

#program to cycle through a group of targets
proc PINES_group {group_filename start_num} {

    puts "STARTING PINES_group..."

    set pines_date [PINES_get_date]

    #set group_file_name "group.txt"

    puts [concat "Reading group coordinates from" /mimir/data/obs72/$pines_date/$group_filename]
    set fid [open /mimir/data/obs72/$pines_date/$group_filename r]
    
    set index 0
    while { [gets $fid line] >= 0 } {
	set list [split $line " "]

	lappend target_list [lindex $list 0]	
	lappend RAhrs_list [lindex $list 1]
	lappend RAmin_list [lindex $list 2]
	lappend RAsec_list [lindex $list 3]
	lappend DECdeg_list [lindex $list 4]
	lappend DECmin_list [lindex $list 5]
	lappend DECsec_list [lindex $list 6]
	lappend exptime_list [lindex $list 7]	
	lappend index_list $index

	incr index
    }
    close $fid

    #Start loop through targets in group
    foreach index $index_list {

	if {$index < $start_num} {continue}

	set target [lindex $target_list $index]
	set exptime [lindex $exptime_list $index]
	
	set RAhrs [lindex $RAhrs_list $index]
	set RAmin [lindex $RAmin_list $index]
	set RAsec [lindex $RAsec_list $index]

	set DECdeg [lindex $DECdeg_list $index]
	set DECmin [lindex $DECmin_list $index]
	set DECsec [lindex $DECsec_list $index]
	
	#make sure the telescope can move to the coordinates
	#covert sexigessimal to decimal
	set ra_deg [expr [expr [scan $RAhrs %d]] + [expr [scan $RAmin %d]] / 60.0 + [expr [scan $RAsec %d]] / 3600.0]
	#for dec, only use degree to avoid sign issues with min and sec
	set dec_deg [expr [scan $DECdeg %d]]

	#calculate hour angle
	set ha [expr [PINES_lst] - $ra_deg]

	if {[expr abs($ha)] > 4.25
	} then { puts [concat "Hour Angle of" $ha "greater than 4.0. Move not executed."]
	} elseif { $ha < 0.0 && $dec_deg > 35.0
	       } then { puts [concat "Target at Hour Angle of" $ha "and Dec of" $dec_deg " degrees, near a restricted postion. Move not executed."]
	} else { puts [concat "Moving to" $target "at" $RAhrs $RAmin $RAsec $DECdeg $DECmin $DECsec ", HA =" $ha "hours..."]
	    
	    move ra=$RAhrs:$RAmin:$RAsec dec=$DECdeg:$DECmin:$DECsec

	    #Give tele 5 seconds to settle before running peakup
	    after 5000

	    puts "Running PINES_peakup."
	    set successful_peakup [PINES_peakup 1.0]

	    if {$successful_peakup == -1
	    } then {puts "PINES_peakup unsuccessful. STOPPING loop."
		break
	    } else {puts "PINES_peakup successful. Running PINES_guide."}
    
	    PINES_guide $exptime 600
	    
	}
    }
    
   

    puts "PINES_group COMPLETE."

    #Three quarks for muster mark.    
    for {set i 0} {$i < 3} {incr i} {
	puts \a\a
	after 300
    }
    
}


