#!/bin/bash


POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -p)
      SILENT_PARAMS="$SILENT_PARAMS -extra_res_fa $2"
      shift # past argument
      shift # past value
      ;;
    -j)
      SILENT_J="$2"
      shift # past argument
      shift # past value
      ;;
    -j*)
      SILENT_J="${1:2:${#1}-2}"
      shift # past value
      ;;
    @*)
      SILENT_AT="$SILENT_AT $1"
      shift # past value
      ;;
    *)    # unknown option
      POSITIONAL+=("$1") # save it in an array for later
      shift # past argument
      ;;
  esac
done

export SILENT_PARAMS
export SILENT_AT

set -- "${POSITIONAL[@]}" # restore positional parameters

