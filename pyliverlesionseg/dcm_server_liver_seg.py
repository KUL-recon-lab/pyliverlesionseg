#TODO input arg for liver model / receiving IP
import sys
import shutil
import time
import threading
import  queue
import argparse
import socket
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

from pydicom.dataset import FileDataset
from pynetdicom import AE, evt
from pynetdicom.sop_class import CTImageStorage, MRImageStorage, VerificationSOPClass

import pyliverlesionseg as pyls

#----------------------------------------------------------------------------------------------------------------
class LiverSegDicomProcessor:
  """
  minimal dicom server that listens for incoming CT/MR dicoms, triggers CNN liver segmentation, and sends
  output RTstruct back to sender

  Currently we rely on the fact that all 2D dicom files of a dicom series are sent between an
  handle accepted and handle released event.
  After storage of a valid CT/MR dicom series, the handle release event is used to trigger the
  CNN liver segmentation (added to a processing queue).

  Parameters
  ----------

  storage_dir ... pathlib.Path
                  master directory used to temporarely store the incoming dicoms

  processing_dir ... pathlib.Path
                     master directory used to process valid input dicoms

  sending_ip     ... str or None
                     IP of the peer used to send back processed results.
                     If None, the IP of the sender is used.

  sending_port   ... int
                     port of peer used to send back processed rtstructs

  cleanup_process_dir ... bool
                          whether to clean up (delete) all input and output files after processing

  timeout             ... uint
                          number of seconds to wait until checking for new items on the processing queue

  mode_name           ... str
                          abs path of the trained CNN model used for liver segmentation
  """
  def __init__(self, storage_dir         = Path.home() / 'liver_seg_dicom_in', 
                     processing_dir      = Path.home() / 'liver_seg_dicom_process',
                     sending_ip          = None, 
                     sending_port        = 104, 
                     cleanup_process_dir = True, 
                     timeout             = 60,
                     model_name          = None):

    self.storage_dir         = storage_dir.resolve()
    self.processing_dir      = processing_dir.resolve()
    self.sending_ip          = sending_ip 
    self.sending_port        = sending_port 
    self.cleanup_process_dir = cleanup_process_dir
    self.timeout             = timeout

    if model_name is None:
      self.model_name = str(Path(pyls.__file__).parent / 'trained_models' / 'model_unet_ct_mr_liv_seg_resize_1.5mm_med_3_resize_3mm_20201011_dice_loss_val_binary_dice_mean.hdf5')
    else:
      self.model_name = model_name

    self.dcm_log_file = self.storage_dir / 'dicom_process.log'
    self.logger       = pyls.setup_logger(self.dcm_log_file, name = 'dicom_io_logger')

    self.processing_queue    = queue.Queue()
    threading.Thread(target = self.worker, daemon = True).start()

    self.reset_last_storage_information()

  def reset_last_storage_information(self):
    self.last_dcm_storage_dir = None
    self.last_dcm_fname       = None
    self.last_peer_address    = None
    self.last_peer_ae_tile    = None
    self.last_peer_port       = None
    self.last_ds              = None


  # Implement a handler for evt.EVT_C_STORE
  def handle_store(self,event):
    self.reset_last_storage_information()

    """Handle a C-STORE request event."""
  
    # get the IP of the sender
    assoc = threading.current_thread()
    self.last_peer_address = assoc.remote['address']
    self.last_peer_ae_tile = assoc.remote['ae_title']
    self.last_peer_port    = assoc.remote['port']
  
    # get string of series description and remove all non alpha-num characters
    sdesc = ''.join(filter(str.isalnum, event.dataset.SeriesDescription))
  
    self.last_dcm_storage_dir = self.storage_dir / f'{event.dataset.StudyDate}_{event.dataset.PatientID}_{event.dataset.StudyInstanceUID}' / f'{event.dataset.Modality}_{sdesc}_{event.dataset.SeriesInstanceUID}'
    self.last_dcm_storage_dir.mkdir(exist_ok = True, parents = True)
    self.last_dcm_fname = self.last_dcm_storage_dir / f'{event.dataset.SOPInstanceUID}.dcm'
 
    # Save the dataset using the SOP Instance UID as the filename
    self.last_ds = FileDataset(self.last_dcm_fname, event.dataset, file_meta = event.file_meta)
    self.last_ds.save_as(self.last_dcm_fname, write_like_original = False)
  
    # Return a 'Success' status
    return 0x0000
  
  def handle_accepted(self,event):
    self.logger.info('accepted')
    return 0x0000
  
  def handle_released(self,event):
    self.logger.info('released')

    if self.last_dcm_storage_dir is not None:
      self.logger.info('')
      self.logger.info(f'series desc  ..: {self.last_ds.SeriesDescription}')
      self.logger.info(f'series UID   ..: {self.last_ds.SeriesInstanceUID}')
      self.logger.info(f'modality     ..: {self.last_ds.Modality}')
      self.logger.info(f'storage dir  ..: {self.last_dcm_storage_dir}')
      self.logger.info(f'peer address ..: {self.last_peer_address}')
      self.logger.info(f'peer AE      ..: {self.last_peer_ae_tile}')
      self.logger.info(f'peer port    ..: {self.last_peer_port}')    
      self.logger.info('')

      if self.last_ds.Modality == 'CT' or self.last_ds.Modality == 'MR':
        self.logger.info('submitting to processing queue')
        try:
          self.logger.info(f'image input {self.last_dcm_storage_dir}')

          # move input dicom series into process dir
          process_dir = self.processing_dir / f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{self.last_dcm_storage_dir.parent.name}'
          process_dir.mkdir(exist_ok = True, parents = True)
        
          shutil.move(self.last_dcm_storage_dir, process_dir / 'image') 
          self.logger.info(f'moving {self.last_dcm_storage_dir} to {process_dir / "image"}')

          # check if study dir is empty after moving series, and delete if it is
          if not any(Path(self.last_dcm_storage_dir.parent).iterdir()):
            shutil.rmtree(self.last_dcm_storage_dir.parent)
            self.logger.info(f'removed empty dir {self.last_dcm_storage_dir.parent}')

          # submit new processing job to the processing queue
          if self.sending_ip is None:
            peer_address =  self.last_peer_address
          else:
            peer_address = self.sending_ip

          self.processing_queue.put((process_dir, peer_address, self.last_ds.Modality))
          self.logger.info(f'adding to process queue {process_dir}')
          self.logger.info(f'current queue {list(self.processing_queue.queue)}')

        except:
          self.logger.error('submitting processing to queue failed')  

    # reset all information about the last valid storage
    # otherwise an unvalid storage request (e.g. PT) will have the wrong last storage information
    self.reset_last_storage_information()

    return 0x0000

  def handle_echo(self,event):
    """Handle a C-ECHO request event."""
    self.logger.info('echo')
    self.reset_last_storage_information()
    return 0x0000

  def worker(self):
    while True:
      try:
        self.logger.info(f'current queue {list(self.processing_queue.queue)}')
        self.logger.info(f'see log file {str(self.dcm_log_file)}')
        process_dir, peer_address, modality = self.processing_queue.get(timeout = self.timeout)
        process_logger = pyls.setup_logger(process_dir.parent / f'{process_dir.name}.log', 
                                           name = process_dir.name)

        process_logger.info(f'Working on {process_dir}')

        try:
          pyls.cnn_liver_lesion_seg_CT_MR_main(str(process_dir / 'image'), 
                                               None, 
                                               self.model_name,
                                               None, 
                                               seg_liver         = True, 
                                               seg_lesion        = False, 
                                               save_nifti        = True, 
                                               input_nifti       = False, 
                                               Modality          = modality,
                                               logger            = process_logger,
                                               dcm_server_params = (peer_address, self.sending_port))
        except:
          process_logger.error('segmentation failed')


        if self.cleanup_process_dir:
          shutil.rmtree(process_dir)
          process_logger.info(f'removed {process_dir}')

        process_logger.info(f'Finished {process_dir}')

        del process_logger
        self.processing_queue.task_done()
      except queue.Empty:
        pass

#------------------------------------------------------------------------------------------------
def main():
  parser = argparse.ArgumentParser(description = 'dicom server for receiving, processing and sending of CT and MR CNN liver segmentations')
  parser.add_argument('--dcm_storage_dir', default = None,   help = 'storage directory for incoming dicoms')
  parser.add_argument('--dcm_process_dir', default = None,   help = 'processing directory for valid dicoms')
  parser.add_argument('--no_cleanup', action = 'store_true', help = 'do not remove processed dicom files')
  parser.add_argument('--AE', default = 'Liver-Seg', help = 'AE title of dicom server')
  parser.add_argument('--listening_port', default = 11112, type = int, 
                      help = 'port where dicom server is listening')
  parser.add_argument('--sending_ip', default = None, 
           help = 'IP of peer to use for sending of output dicom files. If None results are send back to sender.')
  parser.add_argument('--sending_port', default = 104, type = int, 
                      help = 'port of peer to use for sending of output dicom files')
  parser.add_argument('--model_name', default = None, 
           help = 'absolute path of pretrained model for liver segmentation')

  args = parser.parse_args()
  
  if args.dcm_storage_dir is None:
    storage_dir = Path.home() / 'liver_seg_dicom_in'
  else:
    storage_dir = Path(args.storage_dir)

  if args.dcm_process_dir is None:
    process_dir = Path.home() / 'liver_seg_dicom_process'
  else:
    process_dir = Path(args.process_dir)

  # create storage / process dirs if they don't exist
  if not storage_dir.exists(): 
    storage_dir.mkdir(exist_ok = True, parents = True)

  if not process_dir.exists(): 
    process_dir.mkdir(exist_ok = True, parents = True)

  dcm_listener = LiverSegDicomProcessor(storage_dir         = storage_dir, 
                                        processing_dir      = process_dir,
                                        cleanup_process_dir = (not args.no_cleanup),
                                        sending_ip          = args.sending_ip,
                                        sending_port        = args.sending_port,
                                        model_name          = args.model_name)
  
  handlers = [(evt.EVT_C_STORE,  dcm_listener.handle_store), 
              (evt.EVT_RELEASED, dcm_listener.handle_released),
              (evt.EVT_ACCEPTED, dcm_listener.handle_accepted),
              (evt.EVT_C_ECHO,   dcm_listener.handle_echo)]
  
  # Initialise the Application Entity
  ae = AE(args.AE)
  
  # Support presentation contexts for all storage SOP Classes
  ae.add_supported_context(CTImageStorage)
  ae.add_supported_context(MRImageStorage)
  ae.add_supported_context(VerificationSOPClass)
  
  # Start listening for incoming association requests
  ae.start_server((socket.gethostbyname(socket.gethostname()), args.listening_port), 
                   evt_handlers = handlers, block = True)

#------------------------------------------------------------------------------------------------
if __name__ == '__main__':
  main()
